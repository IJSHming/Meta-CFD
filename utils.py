import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import aerosandbox as asb
import matplotlib.pyplot as plt
import numpy as np


def get_kulfan_parameters_from_name(airfoil_name: str) -> np.ndarray:
    """使用 AeroSandbox 将经典翼型转为 18 维 Kulfan 参数"""
    af = asb.KulfanAirfoil(airfoil_name)
    theta = np.concatenate(
        [
            np.array(af.upper_weights, dtype=np.float32),
            np.array(af.lower_weights, dtype=np.float32),
            np.array([af.leading_edge_weight], dtype=np.float32),
            np.array([af.TE_thickness], dtype=np.float32),
        ]
    )
    return theta


def get_kulfan_bounds(init_theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """获取 18 维 Kulfan 参数合理的搜索空间上下界约束"""
    default_lb = np.array([-0.2] * 8 + [-0.6] * 8 + [0.0, 0.0025], dtype=np.float64)
    default_ub = np.array([0.6] * 8 + [0.2] * 8 + [0.5, 0.03], dtype=np.float64)

    lb = np.minimum(init_theta - 0.2, default_lb)
    ub = np.maximum(init_theta + 0.2, default_ub)
    lb[17] = max(0.0025, float(lb[17]))
    return lb, ub


def plot_test_convergence(
    vanilla_calls: np.ndarray,
    vanilla_f_means: np.ndarray,
    meta_calls: np.ndarray,
    meta_f_means: np.ndarray,
    output_file: str = None,
):
    """绘制 Vanilla TuRBO 与 LLM-Meta-TuRBO 真实平均收敛曲线对比"""
    plt.figure(figsize=(9, 5))
    plt.plot(
        vanilla_calls,
        vanilla_f_means,
        label="Vanilla TuRBO (Mean)",
        color="gray",
        linewidth=2.0,
    )
    plt.plot(
        meta_calls,
        meta_f_means,
        label="LLM-Meta-TuRBO (Mean)",
        color="blue",
        linewidth=2.5,
    )
    plt.title("TuRBO Optimization Convergence (Averaged over 10 Runs)")
    plt.xlabel("CFD Call Budget")
    plt.ylabel("Best L/D Ratio Found")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300)
    plt.show()


def plot_airfoil_shapes_and_print_metrics(
    evaluator,
    init_theta: np.ndarray,
    vanilla_theta: np.ndarray,
    meta_theta: np.ndarray,
    airfoil_name: str,
    output_file: str = None,
    output_txt: str = None,
):
    """控制台输出气动参数并绘制翼型几何形状对比（Vanilla vs LLM-Meta）"""
    p_init = evaluator.evaluate_detailed(init_theta)
    p_vanilla = evaluator.evaluate_detailed(vanilla_theta)
    p_meta = evaluator.evaluate_detailed(meta_theta)

    metrics_text = [
        "=" * 65,
        f" Performance Comparison for Test Condition ({airfoil_name.upper()})",
        "=" * 65,
        f"[Initial Airfoil  ({airfoil_name.upper():^8})] L/D: {p_init['ld']:7.2f} | CL: {p_init['cl']:6.4f} | CD: {p_init['cd']:7.5f} | t_max: {p_init['t_max'] * 100:5.2f}%",
        f"[Best Vanilla TuRBO         ] L/D: {p_vanilla['ld']:7.2f} | CL: {p_vanilla['cl']:6.4f} | CD: {p_vanilla['cd']:7.5f} | t_max: {p_vanilla['t_max'] * 100:5.2f}%",
        f"[Best LLM-Meta-TuRBO        ] L/D: {p_meta['ld']:7.2f} | CL: {p_meta['cl']:6.4f} | CD: {p_meta['cd']:7.5f} | t_max: {p_meta['t_max'] * 100:5.2f}%",
        "=" * 65,
    ]
    print("\n" + "\n".join(metrics_text) + "\n")

    if output_txt:
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(metrics_text) + "\n")

    plt.figure(figsize=(10, 4))
    x = np.linspace(0, 1, 1000)

    init_af = p_init["kulfan_af"]
    vanilla_af = p_vanilla["kulfan_af"]
    meta_af = p_meta["kulfan_af"]

    plt.plot(
        x,
        init_af.upper_coordinates(x)[:, 1],
        "k:",
        label=f"Initial ({airfoil_name.upper()})",
    )
    plt.plot(x, init_af.lower_coordinates(x)[:, 1], "k:")

    plt.plot(
        x,
        vanilla_af.upper_coordinates(x)[:, 1],
        "gray",
        linestyle="--",
        label="Best Vanilla TuRBO",
    )
    plt.plot(x, vanilla_af.lower_coordinates(x)[:, 1], "gray", linestyle="--")

    plt.plot(
        x,
        meta_af.upper_coordinates(x)[:, 1],
        "b-",
        linewidth=2.0,
        label="Best LLM-Meta-TuRBO",
    )
    plt.plot(x, meta_af.lower_coordinates(x)[:, 1], "b-", linewidth=2.0)

    plt.title(f"Optimized Airfoil Shape Comparison ({airfoil_name.upper()})")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.gca().set_aspect("equal")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300)
    plt.show()
