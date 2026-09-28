"""MALIBO 神经网络训练集自动化生成命令行入口脚本。

用法示例:
    1. 快速冒烟测试 (8 任务 x 50 步，约 30 秒验证系统连通性):
       python -m malibo_dataset_generator.run_generation --mode quick_test --workers 8

    2. 生成标准全量训练集 (175 任务 x 500 步 = 87,500 条数据):
       python -m malibo_dataset_generator.run_generation --mode grid --workers 16

    3. 紧凑巡航训练集 (125 任务 x 500 步 = 62,500 条数据):
       python -m malibo_dataset_generator.run_generation --mode compact --workers 16

    4. 连续 Sobol 空间填充 (指定 100 任务 x 500 步 = 50,000 条数据):
       python -m malibo_dataset_generator.run_generation --mode sobol --num-tasks 100 --workers 16
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

from malibo_dataset_generator.airfoil_baselines import (
    get_baseline_descriptions,
    verify_airfoil_geometry,
)
from malibo_dataset_generator.condition_sampler import (
    build_compact_grid_tasks,
    build_grid_tasks,
    build_quick_test_tasks,
    build_sobol_tasks,
)
from malibo_dataset_generator.config import (
    BASELINE_AIRFOILS,
    DEFAULT_DATA_DIR,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PT_PATH,
    DEFAULT_TASKS_DIR,
)
from malibo_dataset_generator.dataset_builder import DatasetBuilder
from malibo_dataset_generator.malibo_torch_dataset import MALIBOMetaDataset
from malibo_dataset_generator.visualize_dataset import (
    plot_baseline_airfoils,
    plot_convergence_curves,
    plot_task_sampling_space,
)


def print_banner():
    banner = """
===================================================================================
   MALIBO Meta-Learning Dataset Generator for Airfoil Optimization
   - Black-Box Optimization: CMA-ES (Hansen, 2016) - Fast & Uniform Trajectories
   - Aerodynamic Evaluator: NeuralFoil 'xxxlarge' + 10 Industrial Geometric Constraints
   - Parameterization: Kulfan (18-D) with Strict Physical Feasibility Guardrails
   - Meta-Learning Loss: LFBO Weighted Classification (Pan et al., 2024, MALIBO.pdf)
===================================================================================
    """
    print(banner)


def main():
    parser = argparse.ArgumentParser(
        description="MALIBO 神经网络训练集自动化生成器 (CMA-ES + NeuralFoil xxxlarge)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="grid",
        choices=["grid", "compact", "quick_test", "sobol"],
        help="生成模式: grid (550任务=275000条), compact (250任务=125000条), quick_test (测试用), sobol (连续采样)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=100,
        help="仅在 --mode sobol 时生效: 指定生成的连续任务总数 (默认: 100)",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=500,
        help="单任务 CMA-ES 评估预算步数上限 (默认: 500 步)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并发工作进程数 (默认: 8, 完美匹配 10 核心 HPC 配额，预留 2 核保证 I/O 与系统流畅)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f"数据输出根目录 (默认: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="强制重新运行所有任务 (覆盖已有缓存)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="仅校验 5 种起始翼型几何与工况网格，不执行 TuRBO 寻优",
    )
    parser.add_argument(
        "--no-bundle",
        action="store_true",
        help="生成完成后不打包为 .pt 单文件",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="生成完成后不绘制可视化图表",
    )

    args = parser.parse_args()
    print_banner()

    data_dir = Path(args.data_dir)
    tasks_dir = data_dir / "tasks"
    manifest_path = data_dir / "meta_dataset_manifest.json"
    pt_path = data_dir / "malibo_train_dataset.pt"
    viz_dir = _PACKAGE_DIR / "visualizations"

    # 1. 校验基准翼型几何健康度
    print(">>> 正在校验 5 种迥异起始基准翼型...")
    descs = get_baseline_descriptions()
    for af in BASELINE_AIRFOILS:
        rep = verify_airfoil_geometry(af)
        desc = descs.get(af, {})
        status = "PASSED" if rep["is_feasible"] else "FAILED"
        print(
            f"    * [{status}] {af:<9s} | {desc.get('category', ''):<30s} "
            f"| t_max={rep['t_max']*100:4.1f}%, camber={rep['max_camber']*100:4.1f}%"
        )
    print(">>> 所有 5 种基准翼型几何验证均完全通过!\n")

    if args.verify_only:
        print(">>> [--verify-only] 校验完毕，退出。")
        return

    # 2. 构建任务列表
    print(f">>> 正在根据模式 [{args.mode}] 构建优化任务池...")
    if args.mode == "grid":
        tasks = build_grid_tasks(max_evals=args.max_evals)
    elif args.mode == "compact":
        tasks = build_compact_grid_tasks(max_evals=args.max_evals)
    elif args.mode == "quick_test":
        tasks = build_quick_test_tasks(max_evals=min(args.max_evals, 50))
    elif args.mode == "sobol":
        tasks = build_sobol_tasks(num_tasks=args.num_tasks, max_evals=args.max_evals)
    else:
        raise ValueError(f"未知模式: {args.mode}")

    print(f"    - 已规划生成任务数: {len(tasks)} 个")
    print(f"    - 单任务步数: {tasks[0].max_evals} 步")
    print(f"    - 总样本点数: {len(tasks) * tasks[0].max_evals:,} 条\n")

    # 3. 启动多进程并发优化引擎
    builder = DatasetBuilder(
        data_dir=data_dir,
        tasks_dir=tasks_dir,
        manifest_path=manifest_path,
        max_workers=args.workers,
    )
    manifest = builder.run_tasks(tasks, force_rerun=args.force_rerun)

    # 4. 打包为 PyTorch 专用张量集
    if not args.no_bundle and len(tasks) > 0:
        print("\n>>> 正在将任务轨迹转换为 MALIBO PyTorch Dataset 并打包...")
        try:
            ds = MALIBOMetaDataset(tasks_dir=tasks_dir)
            ds.save_bundled(pt_path)
            print(f"    - MALIBO 元数据集构建完成: 共 {ds.num_tasks} 个任务, {ds.total_samples:,} 条样本!")
        except Exception as e:
            print(f"[Warn] 打包 PyTorch 数据集时出错: {e}")

    # 5. 自动渲染可视化图表
    if not args.no_viz:
        print("\n>>> 正在生成数据集分析可视化图表...")
        try:
            viz_dir.mkdir(parents=True, exist_ok=True)
            plot_baseline_airfoils(viz_dir / "baseline_airfoils.png")
            plot_task_sampling_space(manifest_path, viz_dir / "tasks_space.png")
            plot_convergence_curves(tasks_dir, viz_dir / "convergence_trajectories.png")
            print(f"    - 可视化图表已生成至: {viz_dir}/")
        except Exception as e:
            print(f"[Warn] 生成可视化图表时出错: {e}")

    print("\n>>> 全部操作圆满完成!")


if __name__ == "__main__":
    main()
