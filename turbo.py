"""基于 BoTorch 的 TuRBO-1 (Trust Region Bayesian Optimization) 高性能算法实现"""

import math
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

warnings.filterwarnings("ignore")

# 默认固定 5 次干预步数 (显著大于 n_init=20，且 <= 161，与每 10 步 GP 重拟合精准对齐)
DEFAULT_META_STEPS = (31, 51, 81, 121, 161)


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
    prob_perturb: Optional[float] = None,
    exploration_weight: float = 1.0,
    dtype: torch.dtype = torch.double,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
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

    # 按概率选择扰动维度 (若未指定，采用默认 20/d)
    if prob_perturb is None:
        prob_perturb = min(20.0 / dim, 1.0)
    prob_perturb = float(np.clip(prob_perturb, 0.05, 1.0))

    mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    if len(ind) > 0:
        mask[ind, torch.randint(0, dim, (len(ind),), device=device)] = True

    # 构建候选集合
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    # 高性能 Thompson Sampling (支持 exploration_weight 调节采样不确定性)
    with torch.no_grad():
        posterior = model.posterior(X_cand)
        mu = posterior.mean.squeeze(-1)
        sigma = posterior.variance.clamp_min(1e-9).sqrt().squeeze(-1)
        samples = mu + float(exploration_weight) * sigma * torch.randn_like(mu)
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
    is_meta: bool = False,
    llm_agent: Optional[Any] = None,
    meta_steps: Sequence[int] = DEFAULT_META_STEPS,
    airfoil_name: str = "NACA2412",
    evaluator: Optional[Any] = None,
) -> Tuple[list, list, float, np.ndarray]:
    """执行单次完整的 TuRBO-1 优化流程，支持 Vanilla 模式与固定步数 LLM-Meta 动态元优化。

    参数:
        objective_func: 气动评估函数 (输入 18 维 theta，输出 L/D 升阻比)
        init_theta: 初始几何参数 (18 维)
        lb: 参数下界
        ub: 参数上界
        max_evals: 最大评估预算 (默认 500)
        n_init: 初始采样点数量 (默认 20)
        seed: 随机种子
        verbose_every: 进度打印间隔
        is_meta: 是否启用 LLM 元优化
        llm_agent: LLMMetaAgent 实例
        meta_steps: 固定 LLM 干预步数序列 (默认 [31, 51, 81, 121, 161])
        airfoil_name: 翼型名称 (用于 LLM Prompt)
        evaluator: AerodynamicEvaluator 实例 (用于提取详细气动特征)

    返回:
        calls_history: 评估步数历史
        f_history: 累积最佳目标值历史
        best_f: 找到的最佳升阻比
        best_theta: 对应的最佳 18 维几何参数
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dtype = torch.double
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dim = len(init_theta)
    lb_tensor = torch.tensor(lb, dtype=dtype, device=device)
    ub_tensor = torch.tensor(ub, dtype=dtype, device=device)

    def to_unit_cube(x: torch.Tensor) -> torch.Tensor:
        return (x - lb_tensor) / (ub_tensor - lb_tensor)

    def from_unit_cube(x: torch.Tensor) -> np.ndarray:
        x_real = lb_tensor + (ub_tensor - lb_tensor) * x
        return x_real.detach().cpu().numpy()

    # 1. 初始采样点生成 (n_init 点，包含基准翼型点与 Sobol 扰动点)
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

    init_ld = Y_train_list[0]
    Y_train = torch.tensor(Y_train_list, dtype=dtype, device=device).unsqueeze(-1)

    fail_tol = 6 if max_evals <= 300 else max(4, dim // 3)

    # 3. 初始化 TuRBO 状态
    state = TurboState(
        dim=dim,
        batch_size=1,
        best_value=current_best_f,
        failure_tolerance=fail_tol,
    )

    model = None
    step_count = len(Y_train)
    restart_count = 0
    stagnation_count = 0

    # 动态超参 (受任务最佳实践与 LLM 控制)
    current_prob_perturb = min(10.0 / dim, 0.6)  # 扰动约 50% 坐标 (9 维)，防范全维跳变
    current_exploration_weight = 1.0

    target_meta_steps_set = set(meta_steps)

    # 4. TuRBO 主优化循环
    while len(Y_train) < max_evals:
        target_step = len(Y_train) + 1
        direct_suggested_x = None

        # 检查是否触发信任域重启 (算法自身收缩重置)
        if state.restart_triggered:
            state = TurboState(
                dim=dim,
                batch_size=1,
                best_value=float(Y_train.max().item()),
                failure_tolerance=fail_tol,
            )
            model = None
            restart_count += 1

        # 归一化 Y
        y_mean = Y_train.mean()
        y_std = Y_train.std()
        if y_std < 1e-6:
            norm_Y = Y_train - y_mean
        else:
            norm_Y = (Y_train - y_mean) / y_std

        # 每 10 步或在重启/初始时拟合一次 GP 超参
        if (model is None) or (step_count % 10 == 0):
            covar_module = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=dim, batch_shape=torch.Size([]))
            )
            try:
                model = SingleTaskGP(X_train, norm_Y, covar_module=covar_module)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)
            except Exception:
                model = SingleTaskGP(X_train, norm_Y, covar_module=covar_module)
        else:
            try:
                model.set_train_data(X_train, norm_Y.squeeze(-1), strict=False)
            except Exception:
                model = SingleTaskGP(X_train, norm_Y)

        # --- LLM 固定步数元优化干预 (在第 31, 51, 81, 121, 161 步触发，共 5 次) ---
        if is_meta and llm_agent is not None and (target_step in target_meta_steps_set):
            # 提取近期窗口数据
            window_len = 10
            window_slice = (
                Y_train_list[-window_len:]
                if len(Y_train_list) >= window_len
                else Y_train_list
            )
            window_best = max(window_slice)
            window_mean = float(np.mean(window_slice))
            window_improvements = sum(
                1
                for j in range(1, len(window_slice))
                if window_slice[j] > window_slice[j - 1]
            )

            # 获取当前最佳翼型的详细气动参数
            cl_val, cd_val, t_max_val = 0.0, 0.0, 0.0
            if evaluator is not None:
                try:
                    detailed = evaluator.evaluate_detailed(best_theta)
                    cl_val = detailed.get("cl", 0.0)
                    cd_val = detailed.get("cd", 0.0)
                    t_max_val = detailed.get("t_max", 0.0)
                except Exception:
                    pass

            total_gain = (
                (current_best_f - init_ld) / (abs(init_ld) + 1e-6) * 100.0
            )

            # 提取当前最佳几何在归一化空间 [0, 1]^18 中的坐标
            best_idx = int(Y_train.argmax().item())
            best_unit_vec = [
                round(float(v), 4) for v in X_train[best_idx].detach().cpu().numpy()
            ]

            state_dict = {
                "airfoil_name": airfoil_name,
                "reynolds": getattr(evaluator, "Re", 2.5e6) if evaluator else 2.5e6,
                "alpha": getattr(evaluator, "alpha", 5.0) if evaluator else 5.0,
                "step": target_step,
                "max_evals": max_evals,
                "budget_left": max_evals - len(Y_train),
                "dim": dim,
                "init_ld": init_ld,
                "best_ld": current_best_f,
                "total_gain": total_gain,
                "best_cl": cl_val,
                "best_cd": cd_val,
                "best_t_max": t_max_val,
                "best_point_unit": best_unit_vec,
                "window_best_ld": window_best,
                "window_mean_ld": window_mean,
                "window_improvements": window_improvements,
                "stagnation_count": stagnation_count,
                "tr_length": state.length,
                "tr_min": state.length_min,
                "tr_max": state.length_max,
                "success_counter": state.success_counter,
                "success_tolerance": state.success_tolerance,
                "failure_counter": state.failure_counter,
                "failure_tolerance": state.failure_tolerance,
                "restart_count": restart_count,
            }

            action = llm_agent.decide_action(state_dict)

            # 动态应用 LLM 元动作
            state.length = float(
                np.clip(
                    state.length * action["length_scale_factor"],
                    state.length_min,
                    state.length_max,
                )
            )
            current_prob_perturb = action["prob_perturb"]
            current_exploration_weight = action["exploration_weight"]

            # 检查 LLM 是否直接指定了下一个候选点
            suggested_pt = action.get("suggested_point") or action.get("direct_point")
            has_direct_point = False
            if isinstance(suggested_pt, (list, tuple)) and len(suggested_pt) > 0:
                best_unit_tensor = X_train[best_idx].clone()
                pt_clamped = []
                for dim_idx in range(dim):
                    if dim_idx < len(suggested_pt) and suggested_pt[dim_idx] is not None:
                        try:
                            val = float(suggested_pt[dim_idx])
                            pt_clamped.append(max(0.0, min(1.0, val)))
                        except (ValueError, TypeError):
                            pt_clamped.append(float(best_unit_tensor[dim_idx].item()))
                    else:
                        pt_clamped.append(float(best_unit_tensor[dim_idx].item()))
                direct_suggested_x = torch.tensor([pt_clamped], dtype=dtype, device=device)
                has_direct_point = True

            print(
                f"        [LLM Meta-Intervention @ Step {target_step:3d}] TR Factor: {action['length_scale_factor']:.2f}, Perturb: {current_prob_perturb:.2f}, Explr: {current_exploration_weight:.2f}, DirectPoint: {has_direct_point}"
            )
            if action.get("thought"):
                print(f"          Thought: {action['thought']}")
            if has_direct_point:
                print(f"          --> [Candidate Override] Using LLM directly suggested point for next evaluation!")

        # 生成候选并采样 (若 LLM 直接指定了候选点则直接选用)
        if direct_suggested_x is not None:
            X_next = direct_suggested_x
        else:
            n_candidates = min(100 * dim, 5000)
            X_next = generate_batch(
                state=state,
                model=model,
                X=X_train,
                Y=norm_Y,
                n_candidates=n_candidates,
                prob_perturb=current_prob_perturb,
                exploration_weight=current_exploration_weight,
                dtype=dtype,
                device=device,
            )

        # 评估目标函数
        theta_next = from_unit_cube(X_next[0])
        y_next_val = objective_func(theta_next)
        Y_next = torch.tensor([[y_next_val]], dtype=dtype, device=device)
        Y_train_list.append(y_next_val)

        # 更新状态与历史
        state = update_state(state, Y_next)
        X_train = torch.cat([X_train, X_next], dim=0)
        Y_train = torch.cat([Y_train, Y_next], dim=0)
        step_count = len(Y_train)

        if y_next_val > current_best_f:
            current_best_f = y_next_val
            best_theta = theta_next.copy()
            stagnation_count = 0
        else:
            stagnation_count += 1

        calls_history.append(step_count)
        f_history.append(current_best_f)

        if verbose_every > 0 and step_count % verbose_every == 0 and not is_meta:
            print(
                f"        [CFD 步数 {step_count:3d}/{max_evals}] 当前最优 L/D: {current_best_f:.2f} | 信任域长度: {state.length:.4f}"
            )

    return calls_history, f_history, current_best_f, best_theta
