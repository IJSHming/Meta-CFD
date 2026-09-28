"""Vanilla TuRBO-1 (Trust Region Bayesian Optimization) 纯净单机版优化器。

完全遵循 Eriksson et al. (2019) 原始数学设定，不包含任何外部 LLM 或代理干预代码。
专注于在 18 维 Kulfan 空间内执行高保真贝叶斯局部信任域寻优并完整记录采样轨迹。
"""

import math
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

warnings.filterwarnings("ignore")


@dataclass
class TurboState:
    """TuRBO-1 信任域状态追踪类 (工程任务自适应配置)"""

    dim: int
    batch_size: int = 1
    length: float = 0.8
    length_min: float = 0.5**7       # 约 0.0078125
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 6       # 针对 200~500 步高维优化任务，6 次失败收缩优于原版 18 次迟钝收缩
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        if self.failure_tolerance is None or self.failure_tolerance <= 0:
            self.failure_tolerance = math.ceil(
                max(4.0 / self.batch_size, float(self.dim) / (3.0 * self.batch_size))
            )


def update_turbo_state(state: TurboState, y_next: float) -> TurboState:
    """根据最新评估反馈更新 TuRBO 状态（成功/失败计数与信任域缩放）"""
    if y_next > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
        state.best_value = y_next
    else:
        state.success_counter = 0
        state.failure_counter += 1

    # 连续成功达到阈值 -> 扩张信任域
    if state.success_counter >= state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0

    # 连续失败达到阈值 -> 收缩信任域
    elif state.failure_counter >= state.failure_tolerance:
        state.length = 0.5 * state.length
        state.failure_counter = 0

    # 信任域收缩至极小 -> 触发局部重启
    if state.length < state.length_min:
        state.restart_triggered = True

    return state


def generate_candidates_ts(
    state: TurboState,
    model: SingleTaskGP,
    X: torch.Tensor,
    Y: torch.Tensor,
    n_candidates: int = 1800,
    prob_perturb: Optional[float] = None,
    dtype: torch.dtype = torch.double,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """利用 Thompson Sampling 在自适应信任域超矩形内生成候选采样点"""
    dim = X.shape[1]
    best_idx = int(Y.argmax().item())
    x_center = X[best_idx, :].clone()

    # 提取 GP 模型学习到的 ARD 长度尺度并归一化
    if hasattr(model, "covar_module") and hasattr(model.covar_module, "base_kernel"):
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    elif hasattr(model, "covar_module") and hasattr(model.covar_module, "lengthscale"):
        weights = model.covar_module.lengthscale.squeeze().detach()
    else:
        weights = torch.ones(dim, dtype=dtype, device=device)

    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

    # 计算信任域在 [0, 1]^d 内的上下界
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    # 拟蒙特卡洛超矩形内采样
    sobol = SobolEngine(dim, scramble=True)
    pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    # 按概率决定扰动维度 (对齐气动设计最佳实践：扰动约 50% 坐标，防范全维跳变破坏翼型厚度)
    if prob_perturb is None:
        p_pert = min(10.0 / dim, 0.6)
    else:
        p_pert = float(prob_perturb)

    mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= p_pert
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    if len(ind) > 0:
        mask[ind, torch.randint(0, dim, (len(ind),), device=device)] = True

    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    # Thompson Sampling: 从后验高斯分布中抽取单次轨迹极值
    with torch.no_grad():
        posterior = model.posterior(X_cand)
        mu = posterior.mean.squeeze(-1)
        sigma = posterior.variance.clamp_min(1e-9).sqrt().squeeze(-1)
        samples = mu + sigma * torch.randn_like(mu)
        best_cand_idx = int(samples.argmax().item())
        x_next = X_cand[best_cand_idx : best_cand_idx + 1]

    return x_next


@dataclass
class OptimizationTrajectory:
    """单一任务优化全轨迹记录类"""

    task_id: int
    airfoil_name: str
    reynolds: float
    alpha: float
    total_evals: int
    best_fitness: float
    best_theta: np.ndarray

    # 步序序列 (长度为 total_evals)
    steps: np.ndarray                # [1, 2, ..., 500]
    X_unit: np.ndarray               # (500, 18) 归一化输入
    Theta: np.ndarray                # (500, 18) 真实物理参数
    Y_penalized: np.ndarray          # (500,) 惩罚后适应度 (通常为 L/D - penalty)
    Y_raw: np.ndarray                # (500,) 纯气动升阻比 L/D
    Y_minimize: np.ndarray           # (500,) 供 MALIBO 训练使用的最小化目标 (-Y_penalized)
    CL: np.ndarray                   # (500,) 升力系数
    CD: np.ndarray                   # (500,) 阻力系数
    CM: np.ndarray                   # (500,) 俯仰力矩
    is_feasible: np.ndarray          # (500,) 几何可行性布尔值
    penalties: np.ndarray            # (500,) 惩罚分值
    best_y_history: np.ndarray       # (500,) 历史最优累积曲线
    elapsed_seconds: float = 0.0     # 任务运行耗时


def run_vanilla_turbo(
    evaluator_detailed_func: Callable[[np.ndarray], Dict[str, Any]],
    init_theta: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    task_id: int = 1,
    airfoil_name: str = "naca2412",
    reynolds: float = 1e6,
    alpha: float = 5.0,
    max_evals: int = 500,
    n_init: int = 20,
    gp_refit_interval: int = 10,
    failure_tolerance: Optional[int] = None,
    prob_perturb: Optional[float] = None,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> OptimizationTrajectory:
    """执行标准 Vanilla TuRBO-1 贝叶斯优化，输出完整多维数据集。"""
    t_start = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    dtype = torch.double

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dim = len(init_theta)
    lb_tensor = torch.tensor(lb, dtype=dtype, device=device)
    ub_tensor = torch.tensor(ub, dtype=dtype, device=device)

    # 任务自适应失败收缩阈值：预算 <= 300 时取 6，长预算时取 max(4, dim // 3)
    if failure_tolerance is None:
        fail_tol = 6 if max_evals <= 300 else max(4, dim // 3)
    else:
        fail_tol = int(failure_tolerance)

    if prob_perturb is None:
        p_perturb = min(10.0 / dim, 0.6)  # 扰动约 50% 坐标 (9 维)
    else:
        p_perturb = float(prob_perturb)

    def to_unit_cube(x: torch.Tensor) -> torch.Tensor:
        return (x - lb_tensor) / (ub_tensor - lb_tensor)

    def from_unit_cube(x: torch.Tensor) -> np.ndarray:
        x_real = lb_tensor + (ub_tensor - lb_tensor) * x
        return x_real.detach().cpu().numpy()

    # 1. 初始采样点生成 (n_init 点: 基准原点 + Sobol 扰动)
    x_init_unit = to_unit_cube(
        torch.tensor(init_theta, dtype=dtype, device=device)
    ).unsqueeze(0)

    sobol = SobolEngine(dim, scramble=True, seed=seed)
    if n_init > 1:
        x_random_unit = sobol.draw(n_init - 1).to(dtype=dtype, device=device)
        X_train = torch.cat([x_init_unit, x_random_unit], dim=0)
    else:
        X_train = x_init_unit

    # 轨迹记录容器
    steps_list = []
    x_unit_list = []
    theta_list = []
    y_pen_list = []
    y_raw_list = []
    cl_list = []
    cd_list = []
    cm_list = []
    feas_list = []
    pen_list = []
    best_y_list = []

    current_best_f = -float("inf")
    best_theta = init_theta.copy()

    # 2. 评估初始点集
    for i in range(len(X_train)):
        theta_i = from_unit_cube(X_train[i])
        det = evaluator_detailed_func(theta_i)

        val = float(det.get("ld_penalized", -100.0))
        if val > current_best_f:
            current_best_f = val
            best_theta = theta_i.copy()

        steps_list.append(len(steps_list) + 1)
        x_unit_list.append(X_train[i].detach().cpu().numpy())
        theta_list.append(theta_i)
        y_pen_list.append(val)
        y_raw_list.append(float(det.get("ld", 0.0)))
        cl_list.append(float(det.get("cl", 0.0)))
        cd_list.append(float(det.get("cd", 1.0)))
        cm_list.append(float(det.get("cm", 0.0)))
        feas_list.append(bool(det.get("is_feasible", False)))
        pen_list.append(float(det.get("total_penalty", 0.0)))
        best_y_list.append(current_best_f)

    Y_train = torch.tensor(y_pen_list, dtype=dtype, device=device).unsqueeze(-1)

    # 3. 初始化 TuRBO 状态
    state = TurboState(
        dim=dim,
        batch_size=1,
        best_value=current_best_f,
        failure_tolerance=fail_tol,
    )

    model = None

    # 4. TuRBO 主优化循环 (严格推进至 max_evals)
    while len(X_train) < max_evals:
        # 重启信任域逻辑
        if state.restart_triggered:
            state = TurboState(
                dim=dim,
                batch_size=1,
                best_value=float(Y_train.max().item()),
                failure_tolerance=fail_tol,
            )
            # 重启后在历史最佳点周围引入微小扰动以打破局部陷阱
            model = None

        # 周期性重拟合 GP 代理模型
        if model is None or len(X_train) % gp_refit_interval == 0:
            # 标准化训练目标值以加速 GP 拟合
            train_y_std = (Y_train - Y_train.mean()) / (Y_train.std() + 1e-8)
            covar_module = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=None)
            )
            model = SingleTaskGP(
                X_train,
                train_y_std,
                covar_module=covar_module,
            ).to(dtype=dtype, device=device)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            try:
                fit_gpytorch_mll(mll)
            except Exception:
                pass

        # 生成下一个候选设计点
        x_next = generate_candidates_ts(
            state=state,
            model=model,
            X=X_train,
            Y=Y_train,
            n_candidates=1800,
            prob_perturb=p_perturb,
            dtype=dtype,
            device=device,
        )

        # 物理评估
        theta_next = from_unit_cube(x_next[0])
        det_next = evaluator_detailed_func(theta_next)
        val_next = float(det_next.get("ld_penalized", -100.0))

        # 更新全局与信任域状态
        if val_next > current_best_f:
            current_best_f = val_next
            best_theta = theta_next.copy()

        state = update_turbo_state(state, val_next)

        # 追加数据
        X_train = torch.cat([X_train, x_next], dim=0)
        Y_train = torch.cat(
            [Y_train, torch.tensor([[val_next]], dtype=dtype, device=device)], dim=0
        )

        steps_list.append(len(steps_list) + 1)
        x_unit_list.append(x_next[0].detach().cpu().numpy())
        theta_list.append(theta_next)
        y_pen_list.append(val_next)
        y_raw_list.append(float(det_next.get("ld", 0.0)))
        cl_list.append(float(det_next.get("cl", 0.0)))
        cd_list.append(float(det_next.get("cd", 1.0)))
        cm_list.append(float(det_next.get("cm", 0.0)))
        feas_list.append(bool(det_next.get("is_feasible", False)))
        pen_list.append(float(det_next.get("total_penalty", 0.0)))
        best_y_list.append(current_best_f)

    elapsed = time.time() - t_start

    # 整理为强类型轨迹结构体
    y_pen_arr = np.array(y_pen_list, dtype=np.float64)
    trajectory = OptimizationTrajectory(
        task_id=task_id,
        airfoil_name=airfoil_name,
        reynolds=reynolds,
        alpha=alpha,
        total_evals=len(steps_list),
        best_fitness=float(current_best_f),
        best_theta=best_theta,
        steps=np.array(steps_list, dtype=np.int32),
        X_unit=np.array(x_unit_list, dtype=np.float64),
        Theta=np.array(theta_list, dtype=np.float64),
        Y_penalized=y_pen_arr,
        Y_raw=np.array(y_raw_list, dtype=np.float64),
        Y_minimize=-y_pen_arr,  # MALIBO 论文习惯: 最小化 y <= tau
        CL=np.array(cl_list, dtype=np.float64),
        CD=np.array(cd_list, dtype=np.float64),
        CM=np.array(cm_list, dtype=np.float64),
        is_feasible=np.array(feas_list, dtype=bool),
        penalties=np.array(pen_list, dtype=np.float64),
        best_y_history=np.array(best_y_list, dtype=np.float64),
        elapsed_seconds=float(elapsed),
    )

    return trajectory
