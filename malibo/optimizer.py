"""MALIBO 贝叶斯优化器算法实现 (Algorithm 1).

严格遵循 Pan et al. (2024, Algorithm 1 & Section 4) 流程：
1. 初始设计 (Warm-Start): 通过任务无关均值网络 x_0 = argmax_x g_omega(x) 从元知识中直接推荐最优起点；
2. 目标任务自适应: 在线求解 z_MAP 与拉普拉斯精度矩阵，执行 Thompson 采样 z_hat ~ q(z)；
3. 残差修正集成: 结合梯度提升弱分类器构建综合效用 C_GB(x; z_hat) (Eq. 10)；
4. 准蒙特卡洛极大化: 通过 5,120 个 Sobol 候选点快速定位全局极值候选点 x*。
"""

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.quasirandom import SobolEngine

from malibo.adaptation import LaplaceTaskAdapter
from malibo.config import MALIBOConfig
from malibo.models import MALIBOMetaModel
from malibo.residual_booster import MALIBOResidualBooster


class MALIBOOptimizer:
    """MALIBO 贝叶斯优化算法驱动器 (Algorithm 1)."""

    def __init__(
        self,
        meta_model: MALIBOMetaModel,
        dim: int = 18,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
        config: Optional[MALIBOConfig] = None,
        gamma: float = 1.0 / 3.0,
        init_x: Optional[np.ndarray] = None,
        evaluate_init_first: bool = True,
        n_init: Optional[int] = None,
        budget: int = 200,
        geometry_validator: Optional[Callable[[np.ndarray], bool]] = None,
        device: Optional[torch.device] = None,
    ):
        self.meta_model = meta_model
        self.dim = dim
        self.gamma = gamma
        self.init_x = np.asarray(init_x, dtype=np.float64) if init_x is not None else None
        self.evaluate_init_first = evaluate_init_first
        self.config = config if config is not None else MALIBOConfig()
        self.budget = budget if budget is not None else getattr(self.config.optimization, "budget", 200)
        self.geometry_validator = geometry_validator

        # 初始采样点数最佳实践 (10 步足够为任务自适应提供充分初始化)
        self.n_init = n_init if n_init is not None else getattr(self.config.optimization, "n_init", 10)

        # 自适应信任域核心超参 (平滑重启 + 紧凑微调，杜绝 100x 剧烈跳跃破坏几何)
        self.tr_len = getattr(self.config.optimization, "tr_len_init", 0.4)
        self.tr_len_restart = getattr(self.config.optimization, "tr_len_restart", 0.35)
        self.tr_len_min = getattr(self.config.optimization, "tr_len_min", 0.5**7)
        self.tr_len_max = getattr(self.config.optimization, "tr_len_max", 1.0)
        self.success_tolerance = getattr(self.config.optimization, "success_tolerance", 3)
        self.failure_tolerance = getattr(self.config.optimization, "failure_tolerance", 5)
        self.prob_perturb = getattr(self.config.optimization, "prob_perturb", 0.4)
        self.gamma_init = getattr(self.config.optimization, "gamma_init", 1.0 / 3.0)
        self.gamma_min = getattr(self.config.optimization, "gamma_min", 0.06)
        self.exploration_floor = getattr(self.config.optimization, "exploration_floor", 0.15)
        self.success_counter = 0
        self.failure_counter = 0

        if device is None:
            self.device = next(meta_model.parameters()).device
        else:
            self.device = device

        self.meta_model.eval()
        self.meta_model.to(self.device)

        # 物理边界映射
        self.lb = np.asarray(lb, dtype=np.float64) if lb is not None else np.zeros(dim)
        self.ub = np.asarray(ub, dtype=np.float64) if ub is not None else np.ones(dim)

        # 组件初始化
        self.adapter = LaplaceTaskAdapter(
            meta_model=self.meta_model,
            config=self.config.adaptation,
            gamma=self.gamma,
            device=self.device,
        )
        self.residual_booster = MALIBOResidualBooster(
            meta_model=self.meta_model,
            config=self.config.booster,
            device=self.device,
        )

        # 在线观测数据记录容器
        self.X_unit: List[np.ndarray] = []     # (N, dim) in [0, 1]^dim
        self.Theta: List[np.ndarray] = []      # (N, dim) in real space [lb, ub]
        self.Y_raw: List[float] = []           # (N,) 原始适应度 (如升阻比 L/D，越大越好)
        self.Y_min: List[float] = []           # (N,) 最小化目标 y = -raw (越小越好)
        self.best_y_history: List[float] = []

        self.best_raw_y = -float("inf")
        self.best_theta: Optional[np.ndarray] = None
        self.best_x_unit: Optional[np.ndarray] = self.init_x.copy() if self.init_x is not None else None

    def to_real(self, x_unit: np.ndarray) -> np.ndarray:
        """从 [0, 1]^dim 映射至物理空间 [lb, ub]."""
        return self.lb + (self.ub - self.lb) * x_unit

    def to_unit(self, theta_real: np.ndarray) -> np.ndarray:
        """从物理空间 [lb, ub] 归一化至 [0, 1]^dim."""
        return (theta_real - self.lb) / (self.ub - self.lb + 1e-8)

    def _sample_sobol_candidates(self, n_candidates: int, seed: Optional[int] = None) -> np.ndarray:
        """在搜索空间内生成候选采样集.

        针对高维几何优化 (如 18 维 Kulfan 翼型流形):
        - 围绕当前最佳设计点 best_x 构建自适应信任域超矩形；
        - 采用坐标掩码微扰 (prob_perturb ~ 0.4)，在保护整体气动构型的同时进行高效维度搜索。
        """
        engine = SobolEngine(self.dim, scramble=True, seed=seed)
        x_center = self.best_x_unit if self.best_x_unit is not None else (
            self.init_x if self.init_x is not None else np.full(self.dim, 0.5)
        )

        tr_lb = np.clip(x_center - self.tr_len / 2.0, 0.0, 1.0)
        tr_ub = np.clip(x_center + self.tr_len / 2.0, 0.0, 1.0)

        pert = engine.draw(n_candidates).numpy()
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # 坐标掩码微扰
        prob_pert = self.prob_perturb
        mask = np.random.rand(n_candidates, self.dim) <= prob_pert
        zero_rows = np.where(mask.sum(axis=1) == 0)[0]
        if len(zero_rows) > 0:
            mask[zero_rows, np.random.randint(0, self.dim, len(zero_rows))] = True

        cands = np.tile(x_center, (n_candidates, 1))
        cands[mask] = pert[mask]
        return cands

    def ask(self, n_candidates: Optional[int] = None) -> np.ndarray:
        """生成并推荐下一个采样设计点 x* in [0, 1]^dim (Algorithm 1 lines 1~7)."""
        if n_candidates is None:
            n_candidates = self.config.optimization.n_candidates

        N = len(self.X_unit)

        # 1. 初始设计 (前 n_init 步探索):
        if N == 0:
            if self.init_x is not None and self.evaluate_init_first:
                return np.asarray(self.init_x, dtype=np.float64).copy()

            cands = self._sample_sobol_candidates(n_candidates, seed=self.config.optimization.seed)
            if self.init_x is not None:
                cands[0] = self.init_x

            t_cands = torch.as_tensor(cands, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                utilities = self.meta_model.predict_mean_utility(t_cands).detach().cpu().numpy().flatten()

            best_cand_idx = int(np.argmax(utilities))
            return cands[best_cand_idx]

        if N < self.n_init:
            engine = SobolEngine(self.dim, scramble=True, seed=self.config.optimization.seed + N * 17)
            sobol_point = engine.draw(1).numpy()[0]
            center = self.init_x if self.init_x is not None else np.full(self.dim, 0.5)
            x_cand = np.clip(center + 0.2 * (sobol_point - 0.5), 0.0, 1.0)
            return x_cand

        # 2. 已有充分观测数据: 准备在线输入
        X_arr = np.array(self.X_unit, dtype=np.float32)
        y_min_arr = np.array(self.Y_min, dtype=np.float32)

        t_X = torch.as_tensor(X_arr, dtype=torch.float32, device=self.device)
        t_y = torch.as_tensor(y_min_arr, dtype=torch.float32, device=self.device)

        # 3. 动态退火分位数阈值 tau (随着优化推进，将优质阈值收紧至前沿极值，持续激发改进推力)
        gamma_curr = max(
            self.gamma_min,
            self.gamma_init * np.sqrt(max(0.0, 1.0 - float(N) / max(float(self.budget), 1.0))),
        )
        tau = float(np.quantile(y_min_arr, gamma_curr))

        # 4. 任务自适应: 求解 z_MAP 并抽取 Thompson 样本 z_hat (Algorithm 1 lines 4~6)
        z_sample, z_map = self.adapter.adapt_and_sample(
            t_X, t_y, gamma=gamma_curr, exploration_floor=self.exploration_floor
        )

        # 5. 生成候选采样点
        seed = self.config.optimization.seed + N * 37
        cands = self._sample_sobol_candidates(n_candidates, seed=seed)

        # 6. 计算采集函数: 连续 Logit 决策边距 (Eq. 10 / Eq. 21)
        min_tree_samples = max(self.n_init, self.config.booster.initial_thompson_steps)
        if N >= min_tree_samples:
            self.residual_booster.fit_with_early_stopping(
                X=X_arr,
                y_minimize=y_min_arr,
                tau=tau,
                z_sample=z_sample,
            )
            utilities = self.residual_booster.predict_utility(
                X_candidates=cands,
                z_sample=z_sample,
                use_raw_margin=True,
            )
        else:
            t_cands = torch.as_tensor(cands, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                Phi, m = self.meta_model.extract_features(t_cands)
                z_term = (Phi @ z_sample.to(self.device)).squeeze(-1)
                utilities = (m.squeeze(-1) + z_term).detach().cpu().numpy().flatten()

        # 7. 候选点优选与物理非致命前置校验 (杜绝把评估预算浪费在表面自交叉或厚度坍塌的病态翼型上)
        top_cand_order = np.argsort(utilities)[::-1]
        if self.geometry_validator is not None:
            chosen_cand = None
            for idx in top_cand_order[:150]:
                cand_real = self.to_real(cands[idx])
                if self.geometry_validator(cand_real):
                    chosen_cand = cands[idx]
                    break
            if chosen_cand is not None:
                return chosen_cand

        return cands[top_cand_order[0]]

    def tell(self, x_unit: np.ndarray, raw_y: float) -> None:
        """向优化器反馈最新物理评估结果 (Algorithm 1 line 8).

        参数:
            x_unit: (dim,) 归一化设计点 in [0, 1]^dim
            raw_y: 真实目标值 (如升阻比 L/D，越大越好)
        """
        x_unit = np.asarray(x_unit, dtype=np.float64).flatten()
        theta_real = self.to_real(x_unit)
        y_min = -float(raw_y)  # 转为最小化目标 (y = -L/D)

        self.X_unit.append(x_unit)
        self.Theta.append(theta_real)
        self.Y_raw.append(float(raw_y))
        self.Y_min.append(y_min)

        # 更新全局最优解与自适应信任域
        if self.best_theta is None or raw_y > self.best_raw_y + 1e-3 * max(abs(self.best_raw_y), 1.0):
            self.best_raw_y = float(raw_y)
            self.best_theta = theta_real.copy()
            self.best_x_unit = x_unit.copy()
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter >= self.success_tolerance:
            self.tr_len = min(self.tr_len * 1.5, self.tr_len_max)
            self.success_counter = 0
        elif self.failure_counter >= self.failure_tolerance:
            self.tr_len = self.tr_len * 0.5
            self.failure_counter = 0
            if self.tr_len < self.tr_len_min:
                # 信任域收缩至极小 -> 平滑重启 (重置至 0.35 而非 0.8，防止破坏高精度局部特征)
                self.tr_len = self.tr_len_restart

        self.best_y_history.append(self.best_raw_y)

    def optimize(
        self,
        objective_func: Callable[[np.ndarray], float],
        budget: int = 100,
        verbose_every: int = 20,
    ) -> Dict[str, any]:
        """执行完整端到端贝叶斯元优化闭环."""
        self.budget = budget
        t0 = time.time()
        print(f">>> [MALIBOOptimizer] 启动贝叶斯元优化 (预算: {budget} 步, 维度: {self.dim}D, n_init={self.n_init})...")

        for step in range(1, budget + 1):
            # 1. 生成候选点
            x_next_unit = self.ask()

            # 2. 映射至物理尺度并执行评估
            theta_next = self.to_real(x_next_unit)
            raw_val = float(objective_func(theta_next))

            # 3. 反馈观测数据
            self.tell(x_next_unit, raw_val)

            if step % verbose_every == 0 or step == 1 or step == budget:
                elapsed = time.time() - t0
                print(
                    f"Step [{step:3d}/{budget:3d}] "
                    f"| Current L/D: {raw_val:7.2f} "
                    f"| Best L/D Found: {self.best_raw_y:7.2f} "
                    f"| TR length: {self.tr_len:.4f} "
                    f"| Elapsed: {elapsed:5.1f}s"
                )

        total_time = time.time() - t0
        print(f">>> [MALIBOOptimizer] 优化完成! 最佳 L/D: {self.best_raw_y:.2f}, 耗时: {total_time:.1f}s")

        return {
            "best_y": self.best_raw_y,
            "best_theta": self.best_theta,
            "history": np.array(self.best_y_history),
            "evaluations": len(self.Y_raw),
            "elapsed_seconds": total_time,
            "X_unit": np.array(self.X_unit),
            "Theta": np.array(self.Theta),
            "Y_raw": np.array(self.Y_raw),
        }
