import math
import sys
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass
from typing import Callable, Tuple

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
    """TuRBO-1 信任域状态管理类，完全遵循 Eriksson et al. (2019) 默认配置"""

    dim: int
    batch_size: int = 1
    length: float = 0.8
    length_min: float = 0.5**7  # 约 0.0078125
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 18  # max(4, dim) for batch_size=1 (dim=18 -> 18)
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        if self.failure_tolerance is None or self.failure_tolerance <= 0:
            self.failure_tolerance = math.ceil(
                max(4.0 / self.batch_size, float(self.dim) / self.batch_size)
            )


def update_state(state: TurboState, Y_next: torch.Tensor) -> TurboState:
    """根据最新采样的评估值更新 TuRBO 状态（成功/失败计数与信任域缩放）"""
    y_max = float(Y_next.max())
    if y_max > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
        state.best_value = y_max
    else:
        state.success_counter = 0
        state.failure_counter += 1

    # 成功达到阈值 -> 扩张信任域
    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0

    # 失败达到阈值 -> 收缩信任域
    elif state.failure_counter == state.failure_tolerance:
        state.length = 0.5 * state.length
        state.failure_counter = 0

    # 信任域过小 -> 触发重启
    if state.length < state.length_min:
        state.restart_triggered = True

    return state


def generate_batch(
    state: TurboState,
    model: SingleTaskGP,
    X: torch.Tensor,
    Y: torch.Tensor,
    n_candidates: int = 1800,
    dtype: torch.dtype = torch.double,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """在信任域超矩形内生成候选点，并利用高效 Thompson Sampling 选取极值点"""
    dim = X.shape[1]
    x_center = X[Y.argmax(), :].clone()

    # 提取 GP 模型学习到的 ARD 长度尺度并进行几何归一化加权
    if hasattr(model, "covar_module") and hasattr(model.covar_module, "base_kernel"):
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    elif hasattr(model, "covar_module") and hasattr(model.covar_module, "lengthscale"):
        weights = model.covar_module.lengthscale.squeeze().detach()
    else:
        weights = torch.ones(dim, dtype=dtype, device=device)

    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

    # 计算信任域边界 [0, 1]^d
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    # 在信任域内生成扰动候选点
    sobol = SobolEngine(dim, scramble=True)
    pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    # 按概率选择扰动维度
    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    if len(ind) > 0:
        mask[ind, torch.randint(0, dim, (len(ind),), device=device)] = True

    # 构建候选集合
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    # 高性能 Thompson Sampling: 采用边际采样 mu + sigma * eps 避免 1800x1800 联合协方差矩阵分解
    with torch.no_grad():
        posterior = model.posterior(X_cand)
        mu = posterior.mean.squeeze(-1)
        sigma = posterior.variance.clamp_min(1e-9).sqrt().squeeze(-1)
        samples = mu + sigma * torch.randn_like(mu)
        best_cand_idx = int(samples.argmax().item())
        X_next = X_cand[best_cand_idx : best_cand_idx + 1]

    return X_next


def run_turbo_optimization(
    objective_func: Callable[[np.ndarray], float],
    init_theta: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    max_evals: int = 500,
    n_init: int = 20,
    seed: int = 0,
    verbose_every: int = 100,
) -> Tuple[list, list, float, np.ndarray]:
    """执行单次完整的 TuRBO-1 优化流程。

    参数:
        objective_func: 气动评估函数 (输入 18 维 theta，输出 L/D 升阻比)
        init_theta: 初始几何参数 (18 维)
        lb: 参数下界
        ub: 参数上界
        max_evals: 最大评估预算 (默认 500)
        n_init: 初始采样点数量 (默认 20)
        seed: 随机种子
        verbose_every: 进度打印间隔

    返回:
        calls_history: 评估步数历史
        f_history: 累积最佳目标值历史
        best_f: 找到的最佳升阻比
        best_theta: 对应的最佳 18 维几何参数
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dtype = torch.double
    device = torch.device("cpu")

    dim = len(init_theta)
    lb_tensor = torch.tensor(lb, dtype=dtype, device=device)
    ub_tensor = torch.tensor(ub, dtype=dtype, device=device)

    def to_unit_cube(x: torch.Tensor) -> torch.Tensor:
        return (x - lb_tensor) / (ub_tensor - lb_tensor)

    def from_unit_cube(x: torch.Tensor) -> np.ndarray:
        x_real = lb_tensor + (ub_tensor - lb_tensor) * x
        return x_real.detach().cpu().numpy()

    # 1. 初始采样点生成
    x_init_unit = to_unit_cube(
        torch.tensor(init_theta, dtype=dtype, device=device)
    ).unsqueeze(0)

    sobol = SobolEngine(dim, scramble=True, seed=seed)
    if n_init > 1:
        x_random_unit = sobol.draw(n_init - 1).to(dtype=dtype, device=device)
        X_train = torch.cat([x_init_unit, x_random_unit], dim=0)
    else:
        X_train = x_init_unit

    # 2. 评估初始点
    Y_train_list = []
    calls_history = []
    f_history = []
    current_best_f = -float("inf")
    best_theta = init_theta.copy()

    for i in range(len(X_train)):
        theta_i = from_unit_cube(X_train[i])
        val = objective_func(theta_i)
        Y_train_list.append(val)

        if val > current_best_f:
            current_best_f = val
            best_theta = theta_i.copy()

        calls_history.append(len(calls_history) + 1)
        f_history.append(current_best_f)

    Y_train = torch.tensor(Y_train_list, dtype=dtype, device=device).unsqueeze(-1)

    # 3. 初始化 TuRBO 状态
    state = TurboState(
        dim=dim,
        batch_size=1,
        best_value=current_best_f,
        failure_tolerance=dim,
    )

    model = None
    step_count = len(Y_train)

    # 4. TuRBO 主优化循环
    while len(Y_train) < max_evals:
        # 检查是否触发信任域重启
        if state.restart_triggered:
            state = TurboState(
                dim=dim,
                batch_size=1,
                best_value=float(Y_train.max().item()),
                failure_tolerance=dim,
            )
            model = None

        # 归一化 Y
        y_mean = Y_train.mean()
        y_std = Y_train.std()
        if y_std < 1e-6:
            norm_Y = Y_train - y_mean
        else:
            norm_Y = (Y_train - y_mean) / y_std

        # 每 10 步或在重启/初始时拟合一次 GP 超参，平衡拟合质量与执行效率
        if (model is None) or (step_count % 10 == 0):
            covar_module = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=dim, batch_shape=torch.Size([]))
            )
            try:
                model = SingleTaskGP(X_train, norm_Y, covar_module=covar_module)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll, optimizer_kwargs={"maxiter": 20})
            except Exception:
                model = SingleTaskGP(X_train, norm_Y, covar_module=covar_module)
        else:
            # 增量设置训练数据
            try:
                model.set_train_data(X_train, norm_Y.squeeze(-1), strict=False)
            except Exception:
                model = SingleTaskGP(X_train, norm_Y)

        # 生成候选并采样
        n_candidates = min(100 * dim, 5000)
        X_next = generate_batch(
            state=state,
            model=model,
            X=X_train,
            Y=norm_Y,
            n_candidates=n_candidates,
            dtype=dtype,
            device=device,
        )

        # 评估目标函数
        theta_next = from_unit_cube(X_next[0])
        y_next_val = objective_func(theta_next)
        Y_next = torch.tensor([[y_next_val]], dtype=dtype, device=device)

        # 更新状态与历史
        state = update_state(state, Y_next)
        X_train = torch.cat([X_train, X_next], dim=0)
        Y_train = torch.cat([Y_train, Y_next], dim=0)
        step_count = len(Y_train)

        if y_next_val > current_best_f:
            current_best_f = y_next_val
            best_theta = theta_next.copy()

        calls_history.append(step_count)
        f_history.append(current_best_f)

        if verbose_every > 0 and step_count % verbose_every == 0:
            print(
                f"        [CFD 步数 {step_count:3d}/{max_evals}] 当前最优 L/D: {current_best_f:.2f} | 信任域长度: {state.length:.4f}"
            )

    return calls_history, f_history, current_best_f, best_theta
