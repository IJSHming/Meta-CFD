"""MALIBO 贝叶斯优化器算法实现 (Algorithm 1).

严格对齐原论文 Pan et al. (ICML 2024, Section 4, Algorithm 1 & Appendix G)：
- 彻底移除信赖域 (Trust Region)、稀疏坐标掩码 (prob_perturb)、gamma 退火、探索方差地板与免费几何预筛器 (geometry_validator)；
- 1. 初始化协议 (n_init = 15): 与对比算法 (TuRBO / RL-Meta / LLM-Meta) 共享一致的初始设计点集 (基准构型 + 局部微扰)；
- 2. 目标任务拉普拉斯自适应 (Eq. 6-9): 固定 gamma = 1/3，L-BFGS 求解 z_MAP，构建精确负 Hessian 精度矩阵 Sigma_N^-1，抽取纯净 Thompson 样本 z_hat；
- 3. 残差梯度提升集成 (Eq. 10 & Appendix G): LFBO 双标签 70/30 早停残差 GBDT 构建综合效用 C_GB(x; z_hat)；
- 4. 全局随机搜索极大化 (Appendix G): 在完整 [0, 1]^D 空间均匀随机采样 5,120 个全局候选点并极大化 C_GB(x; z_hat)。
"""

import time
from typing import Callable, Dict, List, Optional
import numpy as np
import torch

from malibo.adaptation import LaplaceTaskAdapter
from malibo.config import MALIBOConfig
from malibo.models import MALIBOMetaModel
from malibo.residual_booster import MALIBOResidualBooster


class MALIBOOptimizer:
    """MALIBO 贝叶斯优化算法驱动器 (Strictly Paper-Exact Algorithm 1, No Trust Region)."""

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
        budget: int = 300,
        device: Optional[torch.device] = None,
    ):
        self.meta_model = meta_model
        self.dim = int(dim)
        self.config = config if config is not None else MALIBOConfig()
        self.gamma = float(getattr(self.config.optimization, "gamma", gamma))
        self.init_x = np.asarray(init_x, dtype=np.float64) if init_x is not None else None
        self.evaluate_init_first = bool(evaluate_init_first)
        self.budget = int(budget if budget is not None else self.config.optimization.budget)
        self.n_init = int(n_init if n_init is not None else self.config.optimization.n_init)
        self.n_candidates = int(self.config.optimization.n_candidates)  # Appendix G: 5,120
        self.seed = int(self.config.optimization.seed)
        self.rng = np.random.RandomState(self.seed)

        if device is None:
            self.device = next(meta_model.parameters()).device
        else:
            self.device = device

        self.meta_model.eval()
        self.meta_model.to(self.device)

        self.lb = np.asarray(lb, dtype=np.float64) if lb is not None else np.zeros(self.dim)
        self.ub = np.asarray(ub, dtype=np.float64) if ub is not None else np.ones(self.dim)

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
            seed=self.seed,
        )

        self.X_unit: List[np.ndarray] = []
        self.Theta: List[np.ndarray] = []
        self.Y_raw: List[float] = []
        self.Y_min: List[float] = []
        self.best_y_history: List[float] = []

        self.best_raw_y = -float("inf")
        self.best_theta: Optional[np.ndarray] = None
        self.best_x_unit: Optional[np.ndarray] = self.init_x.copy() if self.init_x is not None else None

    def to_real(self, x_unit: np.ndarray) -> np.ndarray:
        return self.lb + (self.ub - self.lb) * x_unit

    def to_unit(self, theta_real: np.ndarray) -> np.ndarray:
        return (theta_real - self.lb) / (self.ub - self.lb + 1e-12)

    def ask(self, n_candidates: Optional[int] = None) -> np.ndarray:
        """生成下一个采样设计点 x* in [0, 1]^dim (Strictly Algorithm 1, No Trust Region)."""
        if n_candidates is None:
            n_candidates = self.n_candidates

        N = len(self.X_unit)

        # 1. 初始设计 (与 TuRBO / RL-Meta / LLM-Meta 对齐 n_init 协议)
        if N == 0:
            if self.init_x is not None and self.evaluate_init_first:
                return np.asarray(self.init_x, dtype=np.float64).copy()

            # Algorithm 1 line 1: x_0 = argmax_x g_omega(x) over 5,120 global random samples
            cands = self.rng.uniform(0.0, 1.0, size=(n_candidates, self.dim))
            t_cands = torch.as_tensor(cands, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                utilities = self.meta_model.predict_mean_utility(t_cands).detach().cpu().numpy().reshape(-1)
            return cands[int(np.argmax(utilities))]

        if N < self.n_init:
            # 与 RL-Airfoil/src/optimizers/turbo.py 完全一致的物理空间 ±0.03 初始微扰
            x0_real = self.to_real(self.init_x) if self.init_x is not None else 0.5 * (self.lb + self.ub)
            x_cand_real = np.clip(x0_real + self.rng.uniform(-0.03, 0.03, size=self.dim), self.lb, self.ub)
            return np.clip(self.to_unit(x_cand_real), 0.0, 1.0)

        # 2. 对已观测最小化目标截断极端不可行惩罚 (y_min <= 10.0) 并标准化，保持与元训练尺度一致
        X_arr = np.array(self.X_unit, dtype=np.float64)
        y_clean = np.minimum(np.array(self.Y_min, dtype=np.float64), 10.0)
        y_mean = float(np.mean(y_clean))
        y_std_val = max(float(np.std(y_clean)), 1e-4)
        y_norm = (y_clean - y_mean) / y_std_val
        tau_std = float(np.quantile(y_norm, self.gamma))

        t_X = torch.as_tensor(X_arr, dtype=torch.float32, device=self.device)
        t_y = torch.as_tensor(y_norm, dtype=torch.float32, device=self.device)

        # 3. Algorithm 1 lines 4-6: 拉普拉斯自适应 MAP + 精确精度矩阵 + 纯净 Thompson 采样
        z_sample, _z_map = self.adapter.adapt_and_sample(
            t_X, t_y, tau_std=tau_std, rng=self.rng
        )

        # 4. Algorithm 1 line 7 (Eq. 10 & Appendix G): 70/30 验证早停残差梯度提升树
        self.residual_booster.fit_with_early_stopping(
            X=X_arr,
            y_std=y_norm,
            tau_std=tau_std,
            z_sample=z_sample,
            step_seed=int(self.seed * 1009 + N + 1),
        )

        # 5. Appendix G: 全局 [0, 1]^D 上 5,120 个均匀随机采样点极大化采集函数 (无信赖域、无预筛器)
        cands = self.rng.uniform(0.0, 1.0, size=(n_candidates, self.dim))
        utilities = self.residual_booster.predict_utility(
            X_candidates=cands,
            z_sample=z_sample,
            use_raw_margin=True,
        )
        return cands[int(np.argmax(utilities))]

    def tell(self, x_unit: np.ndarray, raw_y: float) -> None:
        """向优化器反馈最新物理评估结果 (Algorithm 1 line 8)."""
        x_unit = np.asarray(x_unit, dtype=np.float64).reshape(-1)
        theta_real = self.to_real(x_unit)
        y_min = -float(raw_y)

        self.X_unit.append(x_unit)
        self.Theta.append(theta_real)
        self.Y_raw.append(float(raw_y))
        self.Y_min.append(y_min)

        if self.best_theta is None or raw_y > self.best_raw_y:
            self.best_raw_y = float(raw_y)
            self.best_theta = theta_real.copy()
            self.best_x_unit = x_unit.copy()

        self.best_y_history.append(self.best_raw_y)

    def optimize(
        self,
        objective_func: Callable[[np.ndarray], float],
        budget: int = 300,
        verbose_every: int = 50,
    ) -> Dict[str, any]:
        """执行完整端到端纯原论文版 MALIBO 优化闭环."""
        self.budget = int(budget)
        t0 = time.time()
        print(f">>> [MALIBOOptimizer (Paper-Exact)] 启动全局优化 (预算: {self.budget} 步, 维度: {self.dim}D, n_init={self.n_init})...")

        for step in range(1, self.budget + 1):
            x_next_unit = self.ask()
            theta_next = self.to_real(x_next_unit)
            raw_val = float(objective_func(theta_next))
            self.tell(x_next_unit, raw_val)

            if step % verbose_every == 0 or step == 1 or step == self.budget:
                elapsed = time.time() - t0
                print(
                    f"Step [{step:3d}/{self.budget:3d}] "
                    f"| Current L/D: {raw_val:7.2f} "
                    f"| Best L/D Found: {self.best_raw_y:7.2f} "
                    f"| Elapsed: {elapsed:5.1f}s",
                    flush=True,
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
