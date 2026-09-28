"""数据集可视化与几何气动分析工具 (Dataset Visualizer)。

支持：
1. 5 种起始基准翼型几何剖面与弯度线对比图；
2. (Re, Alpha) 工况采样空间覆盖分布图；
3. TuRBO 500 步寻优收敛轨迹曲线对比图；
4. 升阻比分布与可行率统计图。
"""

import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# 导入模块依赖
_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import aerosandbox as asb
from malibo_dataset_generator.airfoil_baselines import get_airfoil_parameters
from malibo_dataset_generator.config import (
    BASELINE_AIRFOILS,
    DEFAULT_DATA_DIR,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_TASKS_DIR,
)


def plot_baseline_airfoils(output_file: Optional[Path] = None) -> None:
    """绘制 5 种迥异起始基准翼型的几何剖面与骨架中弧线对比图。"""
    plt.figure(figsize=(12, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    x_sample = np.linspace(0.0, 1.0, 300)

    for idx, name in enumerate(BASELINE_AIRFOILS):
        theta = get_airfoil_parameters(name)
        af = asb.KulfanAirfoil(
            upper_weights=theta[0:8],
            lower_weights=theta[8:16],
            leading_edge_weight=float(theta[16]),
            TE_thickness=0.0025,
        )
        yu = af.upper_coordinates(x_sample)[:, 1]
        yl = af.lower_coordinates(x_sample)[:, 1]
        camber = 0.5 * (yu + yl)

        color = colors[idx % len(colors)]
        t_max = float(af.max_thickness()) * 100
        c_max = float(af.max_camber()) * 100
        label = f"{name.upper()} (t={t_max:.1f}%, c={c_max:.1f}%)"

        plt.plot(x_sample, yu, color=color, linewidth=2.0, label=label)
        plt.plot(x_sample, yl, color=color, linewidth=2.0)
        plt.plot(x_sample, camber, color=color, linestyle="--", alpha=0.6, linewidth=1.2)

    plt.title("MALIBO Meta-Training Baseline Airfoils (Geometry & Camber Lines)", fontsize=13, fontweight="bold")
    plt.xlabel("Chord Station (x/c)", fontsize=11)
    plt.ylabel("Thickness / Camber (y/c)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.axis("equal")
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300)
        print(f">>> 基准翼型几何图已保存至: {output_file}")
    plt.close()


def plot_task_sampling_space(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    output_file: Optional[Path] = None,
) -> None:
    """绘制所有元任务在 (Re, Alpha) 空间内的覆盖分布图。"""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        print(f"[Warn] 未找到 manifest 文件: {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    tasks = meta.get("tasks_summary", [])
    if not tasks:
        return

    airfoils = sorted(list(set(t["airfoil_name"] for t in tasks)))
    color_map = {af: plt.cm.tab10(i) for i, af in enumerate(airfoils)}

    plt.figure(figsize=(10, 6))
    for af in airfoils:
        sub_tasks = [t for t in tasks if t["airfoil_name"] == af]
        re_vals = [t["reynolds"] for t in sub_tasks]
        alpha_vals = [t["alpha"] for t in sub_tasks]
        plt.scatter(
            re_vals,
            alpha_vals,
            label=f"{af.upper()} (N={len(sub_tasks)})",
            color=color_map[af],
            s=50,
            alpha=0.85,
            edgecolors="none",
        )

    plt.xscale("log")
    plt.title("MALIBO Meta-Tasks Operational Space Coverage", fontsize=13, fontweight="bold")
    plt.xlabel("Reynolds Number (Re, Log Scale)", fontsize=11)
    plt.ylabel("Angle of Attack (Alpha, deg)", fontsize=11)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300)
        print(f">>> 工况空间分布图已保存至: {output_file}")
    plt.close()


def plot_convergence_curves(
    tasks_dir: Path = DEFAULT_TASKS_DIR,
    output_file: Optional[Path] = None,
    max_curves_to_plot: int = 30,
) -> None:
    """绘制 TuRBO 500 步优化收敛曲线采样。"""
    tasks_dir = Path(tasks_dir)
    npz_files = sorted(tasks_dir.glob("task_*.npz"))
    if not npz_files:
        return

    plt.figure(figsize=(10, 5.5))
    step_indices = np.linspace(0, len(npz_files) - 1, min(len(npz_files), max_curves_to_plot), dtype=int)

    for idx in step_indices:
        fpath = npz_files[idx]
        data = np.load(fpath)
        steps = data["steps"]
        best_hist = data["best_y_history"]
        af_name = str(data["airfoil_name"])
        re_val = float(data["reynolds"])
        alpha_val = float(data["alpha"])

        plt.plot(
            steps,
            best_hist,
            linewidth=1.5,
            alpha=0.6,
            label=f"{af_name} Re={re_val:.1e} a={alpha_val:+.1f}" if len(step_indices) <= 6 else None,
        )

    plt.title("Vanilla TuRBO-1 500-Step Optimization Convergence Trajectories", fontsize=13, fontweight="bold")
    plt.xlabel("CFD Evaluation Steps (Budget up to 500)", fontsize=11)
    plt.ylabel("Cumulative Best Penalized L/D Ratio", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    if len(step_indices) <= 6:
        plt.legend(loc="lower right", frameon=True, fontsize=9)
    plt.tight_layout()

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300)
        print(f">>> 收敛轨迹对比图已保存至: {output_file}")
    plt.close()


if __name__ == "__main__":
    out_dir = _PACKAGE_DIR / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_baseline_airfoils(out_dir / "baseline_airfoils.png")
    plot_task_sampling_space(DEFAULT_MANIFEST_PATH, out_dir / "tasks_space.png")
    plot_convergence_curves(DEFAULT_TASKS_DIR, out_dir / "convergence_trajectories.png")
