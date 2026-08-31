import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from typing import Sequence

import numpy as np

from evaluator import AerodynamicEvaluator
from llm_agent import LLMMetaAgent
from turbo import DEFAULT_META_STEPS, run_turbo_optimization
from utils import (
    get_kulfan_bounds,
    get_kulfan_parameters_from_name,
    plot_airfoil_shapes_and_print_metrics,
    plot_test_convergence,
)

CFD_BUDGET = 500  # 评估预算


def test_turbo(
    airfoil_name: str,
    reynolds: float,
    alpha: float,
    n_runs: int = 10,
    llm_url: str = "http://127.0.0.1:30000/v1",
    llm_model: str = "Qwen/Qwen3.8-27B-FP8",
    meta_steps: Sequence[int] = DEFAULT_META_STEPS,
    llm_timeout: float = 60.0,
    api_key: str = "EMPTY",
):
    """测试阶段：对指定翼型与工况执行 n_runs 次独立对比实验 (Vanilla vs LLM-Meta-TuRBO)"""
    evaluator = AerodynamicEvaluator(reynolds=reynolds, alpha=alpha)
    init_theta = get_kulfan_parameters_from_name(airfoil_name)
    lb, ub = get_kulfan_bounds(init_theta)

    llm_agent = LLMMetaAgent(
        base_url=llm_url, model=llm_model, api_key=api_key, timeout=llm_timeout
    )

    print("=" * 70)
    print(
        f">>> 在工况 Re={reynolds:.2e}, Alpha={alpha}°, Airfoil={airfoil_name.upper()} 下执行 {n_runs} 次独立对比实验..."
    )
    print(f"    - LLM 接口地址: {llm_agent.endpoint}")
    print(f"    - LLM 模型名称: {llm_agent.model}")
    print(f"    - LLM 固定干预步数: {list(meta_steps)} (优化过程中固定干预 {len(meta_steps)} 次)")
    print("=" * 70)

    vanilla_calls_histories = []
    vanilla_f_histories = []
    vanilla_best_f = -np.inf
    vanilla_best_theta = None

    meta_calls_histories = []
    meta_f_histories = []
    meta_best_f = -np.inf
    meta_best_theta = None

    for i in range(n_runs):
        print(f"\n>>> [Run {i + 1}/{n_runs}] 正在进行第 {i + 1} 轮独立对比测试...")

        # 1. 运行 Vanilla TuRBO 基线 (n_init=10)
        print(f"  [1/2] 运行 Vanilla TuRBO (Seed={i})...")
        c_v, f_v, b_f_v, b_t_v = run_turbo_optimization(
            objective_func=evaluator.evaluate,
            init_theta=init_theta,
            lb=lb,
            ub=ub,
            max_evals=CFD_BUDGET,
            n_init=10,
            seed=i,
            verbose_every=100,
            is_meta=False,
            evaluator=evaluator,
        )
        vanilla_calls_histories.append(c_v)
        vanilla_f_histories.append(f_v)
        if b_f_v > vanilla_best_f:
            vanilla_best_f = b_f_v
            vanilla_best_theta = b_t_v

        # 2. 运行 LLM-Meta-TuRBO (n_init=10, 固定在第 11, 21, 41, 81, 161 步介入)
        print(f"  [2/2] 运行 LLM-Meta-TuRBO (Seed={i}, LLM={llm_model})...")
        c_m, f_m, b_f_m, b_t_m = run_turbo_optimization(
            objective_func=evaluator.evaluate,
            init_theta=init_theta,
            lb=lb,
            ub=ub,
            max_evals=CFD_BUDGET,
            n_init=10,
            seed=i,
            verbose_every=100,
            is_meta=True,
            llm_agent=llm_agent,
            meta_steps=meta_steps,
            airfoil_name=airfoil_name,
            evaluator=evaluator,
        )
        meta_calls_histories.append(c_m)
        meta_f_histories.append(f_m)
        if b_f_m > meta_best_f:
            meta_best_f = b_f_m
            meta_best_theta = b_t_m

        print(
            f"  --> [第 {i + 1} 轮结果] Vanilla 最优 L/D: {b_f_v:7.2f} | LLM-Meta 最优 L/D: {b_f_m:7.2f}"
        )

    # 计算 10 次运行对齐后的均值
    min_len_v = min(len(h) for h in vanilla_f_histories)
    min_len_m = min(len(h) for h in meta_f_histories)

    vanilla_calls_mean = np.mean(
        [c[:min_len_v] for c in vanilla_calls_histories], axis=0
    )
    vanilla_f_means = np.mean(
        [f[:min_len_v] for f in vanilla_f_histories], axis=0
    )

    meta_calls_mean = np.mean(
        [c[:min_len_m] for c in meta_calls_histories], axis=0
    )
    meta_f_means = np.mean(
        [f[:min_len_m] for f in meta_f_histories], axis=0
    )

    # 保存文件名格式与先前积累的输出保持一致
    base_prefix = f"{airfoil_name.lower()}_{reynolds:.1e}_{alpha:.1f}"
    curve_png = f"{base_prefix}_curve.png"
    shape_png = f"{base_prefix}_shape.png"
    txt_output = f"{base_prefix}.txt"

    plot_test_convergence(
        vanilla_calls_mean,
        vanilla_f_means,
        meta_calls_mean,
        meta_f_means,
        output_file=curve_png,
    )
    plot_airfoil_shapes_and_print_metrics(
        evaluator,
        init_theta,
        vanilla_best_theta,
        meta_best_theta,
        airfoil_name,
        output_file=shape_png,
        output_txt=txt_output,
    )
