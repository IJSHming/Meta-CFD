"""CMA-ES (Covariance Matrix Adaptation Evolution Strategy) 单任务优化器。

专为 MALIBO 元学习数据集高通量生成定制：
1. 计算极速：相较于高斯过程 (GP) 的 O(N^3) 协方差拟合，CMA-ES 计算开销几乎可忽略，生成全量数据集耗时大幅缩短；
2. 线性平稳探索：CMA-ES 优化轨迹具备优良的平稳渐进特性（优化曲线接近直线），生成的数据在各适应度区间分布更加均匀，
   大幅改善 LFBO 分位数加权分类器的正负样本梯度质量，显著增益神经网络 (ResFFN) 训练；
3. 空间有界保真：在 18 维归一化空间 [0, 1]^18 实施边界约束处理，完全对齐 NeuralFoil 与 10 项严苛工业几何校验。
"""

import math
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import cma
import numpy as np

warnings.filterwarnings("ignore")


@dataclass
class OptimizationTrajectory:
    """单一任务优化全轨迹记录类 (与 MALIBO 训练流水线 100% 格式对齐)"""

    task_id: int
    airfoil_name: str
    reynolds: float
    alpha: float
    total_evals: int
    best_fitness: float
    best_theta: np.ndarray

    # 步序序列 (长度为 total_evals)
    steps: np.ndarray                # [1, 2, ..., max_evals]
    X_unit: np.ndarray               # (max_evals, 18) 归一化设计输入 in [0, 1]^18
    Theta: np.ndarray                # (max_evals, 18) 真实物理参数 in [lb, ub]
    Y_penalized: np.ndarray          # (max_evals,) 惩罚后适应度 (L/D - penalty)
    Y_raw: np.ndarray                # (max_evals,) 纯气动升阻比 L/D
    Y_minimize: np.ndarray           # (max_evals,) 供 MALIBO 训练使用的最小化目标 (-Y_penalized)
    CL: np.ndarray                   # (max_evals,) 升力系数
    CD: np.ndarray                   # (max_evals,) 阻力系数
    CM: np.ndarray                   # (max_evals,) 俯仰力矩
    is_feasible: np.ndarray          # (max_evals,) 几何可行性布尔值
    penalties: np.ndarray            # (max_evals,) 惩罚分值
    best_y_history: np.ndarray       # (max_evals,) 历史最优累积曲线
    elapsed_seconds: float = 0.0     # 任务运行耗时


def run_cma_es(
    evaluator_detailed_func: Callable[[np.ndarray], Dict[str, Any]],
    init_theta: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    task_id: int = 1,
    airfoil_name: str = "naca2412",
    reynolds: float = 1e6,
    alpha: float = 5.0,
    max_evals: int = 300,
    sigma0: float = 0.20,
    seed: int = 42,
    device: Optional[Any] = None,  # 保留设备参数接口以兼容上层 Worker 调用
) -> OptimizationTrajectory:
    """执行标准 CMA-ES 进化优化，生成严格为 max_evals 步的完整优化轨迹。

    超参数工程最佳实践：
    - 坐标空间：在归一化超矩形 [0, 1]^18 内运行，消除不同 Bernstein 权重尺度差异；
    - 初始步长 sigma0: 默认 0.20 (在 [0, 1] 空间中覆盖基准点周围约 40% 的有效区域)；
    - 种群规模 popsize: CMA 官方最优设定 4 + floor(3 * ln(18)) = 12；
    - 边界约束: 通过 cma 内部 penalty/repair 机制严格限制在 [0, 1]^18 内部；
    - 基准保证: Step 1 严格评估基准原点 (如 NACA2412)，确保初始适应度锚定真实物理 baseline。
    """
    t_start = time.time()
    np.random.seed(seed)

    dim = len(init_theta)
    lb_arr = np.asarray(lb, dtype=np.float64)
    ub_arr = np.asarray(ub, dtype=np.float64)

    # 归一化初值 (限制在 [0.05, 0.95] 内部防止贴边)
    x0 = np.clip((init_theta - lb_arr) / (ub_arr - lb_arr + 1e-8), 0.05, 0.95)

    # 容器初始化
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

    # 1. 优先评估任务基准初值 (Step 1 物理基准锚定)
    det_0 = evaluator_detailed_func(init_theta)
    val_0 = float(det_0.get("ld_penalized", -100.0))
    current_best_f = val_0
    best_theta = init_theta.copy()

    steps_list.append(1)
    x_unit_list.append(x0.copy())
    theta_list.append(init_theta.copy())
    y_pen_list.append(val_0)
    y_raw_list.append(float(det_0.get("ld", 0.0)))
    cl_list.append(float(det_0.get("cl", 0.0)))
    cd_list.append(float(det_0.get("cd", 1.0)))
    cm_list.append(float(det_0.get("cm", 0.0)))
    feas_list.append(bool(det_0.get("is_feasible", False)))
    pen_list.append(float(det_0.get("total_penalty", 0.0)))
    best_y_list.append(current_best_f)

    # 2. 初始化 CMA-ES 实例
    cma_opts = {
        "bounds": [0.0, 1.0],
        "seed": seed,
        "verbose": -9,
        "maxfevals": max_evals,
        "tolfun": 1e-11,
        "tolx": 1e-11,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, cma_opts)

    # 3. CMA-ES 进化循环 (严格推进至 max_evals 步)
    while len(steps_list) < max_evals:
        # 若触发早停条件，则在当前最优解周围重启新一轮进化
        if es.stop():
            best_unit = np.clip((best_theta - lb_arr) / (ub_arr - lb_arr + 1e-8), 0.05, 0.95)
            cma_opts["seed"] = seed + len(steps_list)
            es = cma.CMAEvolutionStrategy(best_unit, sigma0 * 0.75, cma_opts)

        cands = es.ask()
        fitnesses = []

        for cand in cands:
            if len(steps_list) >= max_evals:
                break

            cand_unit = np.clip(cand, 0.0, 1.0)
            theta_i = lb_arr + (ub_arr - lb_arr) * cand_unit

            det = evaluator_detailed_func(theta_i)
            val = float(det.get("ld_penalized", -100.0))

            if val > current_best_f:
                current_best_f = val
                best_theta = theta_i.copy()

            steps_list.append(len(steps_list) + 1)
            x_unit_list.append(cand_unit.copy())
            theta_list.append(theta_i.copy())
            y_pen_list.append(val)
            y_raw_list.append(float(det.get("ld", 0.0)))
            cl_list.append(float(det.get("cl", 0.0)))
            cd_list.append(float(det.get("cd", 1.0)))
            cm_list.append(float(det.get("cm", 0.0)))
            feas_list.append(bool(det.get("is_feasible", False)))
            pen_list.append(float(det.get("total_penalty", 0.0)))
            best_y_list.append(current_best_f)

            # CMA-ES 最小化目标: y_minimize = -val
            fitnesses.append(-val)

        # 仅当完整评估一整代候选解时才反馈给进化策略更新协方差矩阵
        if len(fitnesses) == len(cands):
            es.tell(cands, fitnesses)

    elapsed = float(time.time() - t_start)

    return OptimizationTrajectory(
        task_id=task_id,
        airfoil_name=airfoil_name,
        reynolds=reynolds,
        alpha=alpha,
        total_evals=len(steps_list),
        best_fitness=current_best_f,
        best_theta=best_theta,
        steps=np.array(steps_list, dtype=np.int32),
        X_unit=np.array(x_unit_list, dtype=np.float64),
        Theta=np.array(theta_list, dtype=np.float64),
        Y_penalized=np.array(y_pen_list, dtype=np.float64),
        Y_raw=np.array(y_raw_list, dtype=np.float64),
        Y_minimize=-np.array(y_pen_list, dtype=np.float64),
        CL=np.array(cl_list, dtype=np.float64),
        CD=np.array(cd_list, dtype=np.float64),
        CM=np.array(cm_list, dtype=np.float64),
        is_feasible=np.array(feas_list, dtype=bool),
        penalties=np.array(pen_list, dtype=np.float64),
        best_y_history=np.array(best_y_list, dtype=np.float64),
        elapsed_seconds=elapsed,
    )
