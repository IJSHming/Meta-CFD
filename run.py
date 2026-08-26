import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

from evaluator import AerodynamicEvaluator
from turbo import run_turbo_optimization
from utils import (
    get_kulfan_bounds,
    get_kulfan_parameters_from_name,
    plot_airfoil_shapes_and_print_metrics,
    plot_test_convergence,
)

CFD_BUDGET = 500  # 评估预算


def test_turbo(airfoil_name: str, reynolds: float, alpha: float, n_runs: int = 10):
    """测试阶段：对指定翼型与工况执行 n_runs 次独立 TuRBO 优化实验"""
    evaluator = AerodynamicEvaluator(reynolds=reynolds, alpha=alpha)
    init_theta = get_kulfan_parameters_from_name(airfoil_name)
    lb, ub = get_kulfan_bounds(init_theta)

    print(
        f">>> 在工况 Re={reynolds:.2e}, Alpha={alpha}°, Airfoil={airfoil_name.upper()} 下执行 {n_runs} 次独立 TuRBO 测试实验..."
    )

    calls_histories = []
    f_histories = []
    best_f = -np.inf
    best_theta = None

    for i in range(n_runs):
        print(f"  --> 正在进行第 {i + 1}/{n_runs} 次独立优化实验...")
        c, f, b_f, b_t = run_turbo_optimization(
            objective_func=evaluator.evaluate,
            init_theta=init_theta,
            lb=lb,
            ub=ub,
            max_evals=CFD_BUDGET,
            n_init=20,
            seed=i,
            verbose_every=100,
        )
        calls_histories.append(c)
        f_histories.append(f)
        if b_f > best_f:
            best_f = b_f
            best_theta = b_t
        print(f"      [第 {i + 1} 次完成] 当前最优 L/D: {b_f:.2f}")

    # 计算 10 次运行对齐后的均值
    min_len = min(len(h) for h in f_histories)
    calls_mean = np.mean([c[:min_len] for c in calls_histories], axis=0)
    f_means = np.mean([f[:min_len] for f in f_histories], axis=0)

    # 保存文件名格式与先前积累的输出保持一致
    base_prefix = f"{airfoil_name.lower()}_{reynolds:.1e}_{alpha:.1f}"
    curve_png = f"{base_prefix}_curve.png"
    shape_png = f"{base_prefix}_shape.png"
    txt_output = f"{base_prefix}.txt"

    plot_test_convergence(calls_mean, f_means, output_file=curve_png)
    plot_airfoil_shapes_and_print_metrics(
        evaluator,
        init_theta,
        best_theta,
        airfoil_name,
        output_file=shape_png,
        output_txt=txt_output,
    )
