"""雷诺数与攻角工况采样器与任务描述类 (TaskSpec)。

支持离散结构化网格采样 (Grid Sampling) 与连续准蒙特卡洛空间填充 (Sobol Sampling)。
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from malibo_dataset_generator.airfoil_baselines import (
    get_airfoil_bounds,
    get_airfoil_parameters,
)
from malibo_dataset_generator.config import (
    BASELINE_AIRFOILS,
    COMPACT_ALPHA_GRID,
    DEFAULT_ALPHA_GRID,
    DEFAULT_REYNOLDS_GRID,
    QUICK_AIRFOILS,
    QUICK_ALPHAS,
    QUICK_REYNOLDS,
)


@dataclass
class TaskSpec:
    """单一优化元任务描述规范"""

    task_id: int
    airfoil_name: str
    reynolds: float
    alpha: float
    seed: int
    max_evals: int
    init_theta: np.ndarray
    lb: np.ndarray
    ub: np.ndarray

    @property
    def task_name(self) -> str:
        """任务唯一标准字符串标识"""
        return f"task_{self.task_id:04d}_{self.airfoil_name}_Re{self.reynolds:.1e}_a{self.alpha:+.1f}"

    def to_dict(self) -> dict:
        """转为字典格式以便 JSON 序列化"""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "airfoil_name": self.airfoil_name,
            "reynolds": float(self.reynolds),
            "log_reynolds": float(np.log10(self.reynolds)),
            "alpha": float(self.alpha),
            "seed": int(self.seed),
            "max_evals": int(self.max_evals),
        }


def build_grid_tasks(
    airfoils: Sequence[str] = BASELINE_AIRFOILS,
    reynolds_list: Sequence[float] = DEFAULT_REYNOLDS_GRID,
    alphas_list: Sequence[float] = DEFAULT_ALPHA_GRID,
    max_evals: int = 500,
    base_seed: int = 42,
) -> List[TaskSpec]:
    """构建标准笛卡尔积离散网格任务池。

    默认: 5 翼型 x 10 Re x 11 Alpha = 550 个 Meta-tasks。
    """
    tasks = []
    task_id = 0

    # 预先缓存各翼型的基准参数与边界，避免重复计算
    geom_cache = {}
    for af in airfoils:
        th = get_airfoil_parameters(af)
        lb, ub = get_airfoil_bounds(th)
        geom_cache[af] = (th, lb, ub)

    for af in airfoils:
        th, lb, ub = geom_cache[af]
        for re in reynolds_list:
            for alpha in alphas_list:
                task_id += 1
                seed = base_seed + task_id * 17
                tasks.append(
                    TaskSpec(
                        task_id=task_id,
                        airfoil_name=af,
                        reynolds=float(re),
                        alpha=float(alpha),
                        seed=seed,
                        max_evals=max_evals,
                        init_theta=th.copy(),
                        lb=lb.copy(),
                        ub=ub.copy(),
                    )
                )
    return tasks


def build_compact_grid_tasks(
    airfoils: Sequence[str] = BASELINE_AIRFOILS,
    reynolds_list: Sequence[float] = DEFAULT_REYNOLDS_GRID,
    alphas_list: Sequence[float] = COMPACT_ALPHA_GRID,
    max_evals: int = 500,
    base_seed: int = 42,
) -> List[TaskSpec]:
    """构建紧凑型巡航网格任务池 (5 x 10 x 5 = 250 任务)。"""
    return build_grid_tasks(
        airfoils=airfoils,
        reynolds_list=reynolds_list,
        alphas_list=alphas_list,
        max_evals=max_evals,
        base_seed=base_seed,
    )


def build_quick_test_tasks(
    airfoils: Sequence[str] = QUICK_AIRFOILS,
    reynolds_list: Sequence[float] = QUICK_REYNOLDS,
    alphas_list: Sequence[float] = QUICK_ALPHAS,
    max_evals: int = 50,  # 冒烟测试可只跑 50 步
    base_seed: int = 42,
) -> List[TaskSpec]:
    """构建快速测试任务池 (2 翼型 x 2 Re x 2 Alpha = 8 任务)。"""
    return build_grid_tasks(
        airfoils=airfoils,
        reynolds_list=reynolds_list,
        alphas_list=alphas_list,
        max_evals=max_evals,
        base_seed=base_seed,
    )


def build_sobol_tasks(
    num_tasks: int = 100,
    airfoils: Sequence[str] = BASELINE_AIRFOILS,
    log_re_range: Tuple[float, float] = (4.0, 8.0),
    alpha_range: Tuple[float, float] = (-3.0, 15.0),
    max_evals: int = 500,
    base_seed: int = 42,
) -> List[TaskSpec]:
    """采用 2D Sobol 低差异序列在 (log10(Re), Alpha) 连续空间内生成任务。

    在工况超矩形内实现最优空间填充 (Space-filling)，循环分配至所选基准翼型。
    """
    sobol = SobolEngine(dimension=2, scramble=True, seed=base_seed)
    pts = sobol.draw(num_tasks).numpy()

    # 映射至物理尺度
    log_re_min, log_re_max = log_re_range
    alpha_min, alpha_max = alpha_range

    log_re_vals = log_re_min + pts[:, 0] * (log_re_max - log_re_min)
    re_vals = 10.0 ** log_re_vals
    alpha_vals = alpha_min + pts[:, 1] * (alpha_max - alpha_min)

    # 缓存翼型参数
    geom_cache = {}
    for af in airfoils:
        th = get_airfoil_parameters(af)
        lb, ub = get_airfoil_bounds(th)
        geom_cache[af] = (th, lb, ub)

    tasks = []
    num_airfoils = len(airfoils)
    for i in range(num_tasks):
        af = airfoils[i % num_airfoils]
        th, lb, ub = geom_cache[af]
        task_id = i + 1
        seed = base_seed + task_id * 17
        tasks.append(
            TaskSpec(
                task_id=task_id,
                airfoil_name=af,
                reynolds=float(round(re_vals[i], -1)),
                alpha=float(round(alpha_vals[i], 2)),
                seed=seed,
                max_evals=max_evals,
                init_theta=th.copy(),
                lb=lb.copy(),
                ub=ub.copy(),
            )
        )
    return tasks
