"""MALIBO 贝叶斯优化测试评估脚本 (test_airfoil_bo.py).

在指定翼型与工况下执行 MALIBO 优化，并支持与 Vanilla TuRBO 进行同条件对比评测。

用法示例:
    python -m malibo.test_airfoil_bo --airfoil naca2412 --reynolds 2.5e6 --alpha 5.0 --budget 50 --compare-turbo
"""

import argparse
import sys
import time
from pathlib import Path

# 确保控制台输出 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluator import AerodynamicEvaluator
from malibo.config import MALIBOConfig, ResFFNConfig
from malibo.models import MALIBOMetaModel
from malibo.optimizer import MALIBOOptimizer
from malibo_dataset_generator.airfoil_baselines import (
    get_airfoil_bounds,
    get_airfoil_parameters,
)
from malibo_dataset_generator.vanilla_turbo import run_vanilla_turbo


def main():
    parser = argparse.ArgumentParser(
        description="MALIBO 贝叶斯优化测试与 Vanilla TuRBO 对比脚本"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(_PACKAGE_DIR / "malibo_checkpoint.pt"),
        help="已训练的 MALIBO 权重路径",
    )
    parser.add_argument(
        "--airfoil",
        type=str,
        default="naca2412",
        help="测试基准翼型 (默认: naca2412)",
    )
    parser.add_argument(
        "--reynolds",
        type=float,
        default=2.5e6,
        help="测试雷诺数 Re (默认: 2.5e6)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=5.0,
        help="测试攻角 Alpha (度, 默认: 5.0)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=50,
        help="优化评估步数上限 (默认: 50 步)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="测试随机种子 (默认: 42)",
    )
    parser.add_argument(
        "--compare-turbo",
        action="store_true",
        help="是否同时运行同预算的 Vanilla TuRBO 进行对比",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default=str(_PACKAGE_DIR / "malibo_vs_turbo.png"),
        help="收敛对比曲线输出路径",
    )

    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"未找到检查点文件: {ckpt_path}！请先运行 python -m malibo.train_malibo 训练网络。"
        )

    # 1. 准备物理评估器与基准参数
    evaluator = AerodynamicEvaluator(reynolds=args.reynolds, alpha=args.alpha)
    init_theta = get_airfoil_parameters(args.airfoil)
    lb, ub = get_airfoil_bounds(init_theta)

    det_init = evaluator.evaluate_detailed(init_theta)
    init_ld = float(det_init.get("ld_penalized", -100.0))

    print("=" * 85)
    print(f">>> [MALIBO Test] 测试工况: {args.airfoil.upper()} @ Re={args.reynolds:.2e}, Alpha={args.alpha:+.1f}°")
    print(f"    - 基准初始 L/D: {init_ld:.2f}")
    print(f"    - 优化预算: {args.budget} 步")
    print(f"    - 权重路径: {ckpt_path}")
    print("=" * 85)

    # 2. 加载 MALIBO 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_tasks = ckpt.get("num_tasks", 175)
    feature_dim = ckpt.get("feature_dim", 50)

    res_cfg = ResFFNConfig(in_dim=18, feature_dim=feature_dim)
    model = MALIBOMetaModel(num_tasks=num_tasks, config=res_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    # 3. 运行 MALIBO 元优化
    print("\n--- [1/2] 运行 MALIBO 元学习贝叶斯优化 ---")
    opt_cfg = MALIBOConfig()
    opt_cfg.optimization.seed = args.seed
    opt_cfg.optimization.budget = args.budget
    init_x_unit = (init_theta - lb) / (ub - lb + 1e-8)

    def is_geom_valid(th: np.ndarray) -> bool:
        _, af = evaluator._theta_to_kulfan(th)
        is_fatal, _, _ = evaluator.validate_geometry(af)
        return not is_fatal

    malibo_opt = MALIBOOptimizer(
        meta_model=model,
        dim=18,
        lb=lb,
        ub=ub,
        init_x=init_x_unit,
        evaluate_init_first=True,
        config=opt_cfg,
        budget=args.budget,
        geometry_validator=is_geom_valid,
        device=device,
    )

    malibo_res = malibo_opt.optimize(
        objective_func=evaluator.evaluate,
        budget=args.budget,
        verbose_every=10,
    )

    # 4. 可选: 运行 Vanilla TuRBO 进行对比
    turbo_history = None
    if args.compare_turbo:
        print("\n--- [2/2] 运行 Vanilla TuRBO-1 作为基线对比 ---")
        turbo_traj = run_vanilla_turbo(
            evaluator_detailed_func=evaluator.evaluate_detailed,
            init_theta=init_theta,
            lb=lb,
            ub=ub,
            airfoil_name=args.airfoil,
            reynolds=args.reynolds,
            alpha=args.alpha,
            max_evals=args.budget,
            seed=args.seed,
            device=device,
        )
        turbo_history = turbo_traj.best_y_history
        print(
            f"    - Vanilla TuRBO 最佳 L/D: {turbo_traj.best_fitness:.2f}, 耗时: {turbo_traj.elapsed_seconds:.1f}s"
        )

    # 5. 总结对比结果
    print("\n" + "=" * 85)
    print(">>> 优化结果综合对比报告:")
    print(f"    * 基准初始 L/D:        {init_ld:.2f}")
    print(f"    * MALIBO 最终最佳 L/D:   {malibo_res['best_y']:.2f} (Δ+{malibo_res['best_y'] - init_ld:.2f}) | 耗时: {malibo_res['elapsed_seconds']:.1f}s")
    if turbo_history is not None:
        turbo_best = float(turbo_history[-1])
        print(f"    * Vanilla TuRBO 最佳 L/D: {turbo_best:.2f} (Δ+{turbo_best - init_ld:.2f}) | 耗时: {turbo_traj.elapsed_seconds:.1f}s")
    print("=" * 85)

    # 6. 绘制收敛对比曲线
    if args.output_plot:
        plt.figure(figsize=(9, 5))
        steps = np.arange(1, len(malibo_res["history"]) + 1)
        plt.plot(steps, malibo_res["history"], label=f"MALIBO (Best: {malibo_res['best_y']:.2f})", color="crimson", linewidth=2.2)

        if turbo_history is not None:
            plt.plot(steps, turbo_history, label=f"Vanilla TuRBO (Best: {turbo_best:.2f})", color="steelblue", linestyle="--", linewidth=2.0)

        plt.axhline(init_ld, color="gray", linestyle=":", label=f"Baseline Initial ({init_ld:.1f})")
        plt.title(f"Airfoil Optimization: MALIBO vs TuRBO ({args.airfoil.upper()} Re={args.reynolds:.1e}, a={args.alpha:+.1f}°)", fontsize=12, fontweight="bold")
        plt.xlabel("Evaluation Budget", fontsize=11)
        plt.ylabel("Best Penalized L/D Ratio Found", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower right", frameon=True, fontsize=10)
        plt.tight_layout()

        out_path = Path(args.output_plot)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)
        print(f">>> 对比曲线已保存至: {out_path}")
        plt.close()


if __name__ == "__main__":
    main()
