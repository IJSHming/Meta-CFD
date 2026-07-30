"""辅助工具：Kulfan 参数生成、10*10*5 网格构建、测试运行与绘图/打印"""

import aerosandbox as asb
import matplotlib.pyplot as plt
import numpy as np

from evaluator import AerodynamicEvaluator


def get_kulfan_parameters_from_name(airfoil_name):
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


def build_training_grid():
    """构建 10 (Re) * 10 (Alpha) * 5 (Airfoils) = 500 种组合表格

    - 雷诺数: 5e5 ~ 5e6 (均分 10 个)
    - 攻角: 1.0 ~ 10.0 deg (均分 10 个)
    - 起始形状: NACA 2412, NACA 0012, NACA 0015, NACA 4412, NACA 2415
    """
    re_list = np.linspace(5e5, 5e6, 10)
    alpha_list = np.linspace(1.0, 10.0, 10)
    airfoil_names = ["naca2412", "naca0012", "naca0015", "naca4412", "naca2415"]

    grid = []
    for re in re_list:
        for alpha in alpha_list:
            evaluator = AerodynamicEvaluator(reynolds=re, alpha=alpha)
            for name in airfoil_names:
                init_theta = get_kulfan_parameters_from_name(name)
                grid.append((evaluator, init_theta))
    return grid


def run_evaluation(env, model=None, is_rl=False):
    """单次优化评估推断，返回轨迹与最佳结果"""
    obs, _ = env.reset()
    done = False
    calls_history = [0]
    fnew_history = [env.f]

    best_theta = env.best_theta.copy()
    best_f = env.best_f

    while not done:
        if is_rl and model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.array([0.0, 0.0], dtype=np.float32)

        obs, _, terminated, truncated, info = env.step(action)
        calls_history.append(info["calls"])
        fnew_history.append(info["f_new"])
        if info["best_f"] > best_f:
            best_f = info["best_f"]
            best_theta = info["best_theta"]
        done = terminated or truncated

    return calls_history, fnew_history, best_f, best_theta


def plot_test_convergence(vanilla_calls, vanilla_f_means, rl_calls, rl_f_means):
    """绘制真实平均收敛曲线（无滑动平均）"""
    plt.figure(figsize=(9, 5))
    plt.plot(
        vanilla_calls,
        vanilla_f_means,
        label="Vanilla SPSA (Mean)",
        color="gray",
        linewidth=2.0,
    )
    plt.plot(
        rl_calls,
        rl_f_means,
        label="RL-SPSA (Mean)",
        color="blue",
        linewidth=2.5,
    )
    plt.title("Optimization Convergence Curve (Averaged over 10 Runs)")
    plt.xlabel("CFD Call Budget")
    plt.ylabel("Current L/D Ratio")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_airfoil_shapes_and_print_metrics(
    evaluator, init_theta, vanilla_theta, rl_theta, airfoil_name
):
    """绘制 3 种翼型几何形状对比，并在控制台 print 详细气动参数"""
    p_init = evaluator.evaluate_detailed(init_theta)
    p_vanilla = evaluator.evaluate_detailed(vanilla_theta)
    p_rl = evaluator.evaluate_detailed(rl_theta)

    # 1. 命令行输出气动性能数值
    print("\n" + "=" * 65)
    print(f" Performance Comparison for Test Condition ({airfoil_name.upper()})")
    print("=" * 65)
    print(
        f"[Initial Airfoil  ({airfoil_name.upper():^8})] L/D: {p_init['ld']:7.2f} | CL: {p_init['cl']:6.4f} | CD: {p_init['cd']:7.5f} | t_max: {p_init['t_max'] * 100:5.2f}%"
    )
    print(
        f"[Best Vanilla SPSA          ] L/D: {p_vanilla['ld']:7.2f} | CL: {p_vanilla['cl']:6.4f} | CD: {p_vanilla['cd']:7.5f} | t_max: {p_vanilla['t_max'] * 100:5.2f}%"
    )
    print(
        f"[Best RL-SPSA (SAC)         ] L/D: {p_rl['ld']:7.2f} | CL: {p_rl['cl']:6.4f} | CD: {p_rl['cd']:7.5f} | t_max: {p_rl['t_max'] * 100:5.2f}%"
    )
    print("=" * 65 + "\n")

    # 2. 绘制纯几何翼型对比图（利用 AeroSandbox KulfanAirfoil 实例）
    plt.figure(figsize=(10, 4))
    x = np.linspace(0, 1, 1000)

    init_af = p_init["kulfan_af"]
    van_af = p_vanilla["kulfan_af"]
    rl_af = p_rl["kulfan_af"]

    plt.plot(
        x,
        init_af.upper_coordinates(x)[:, 1],
        "k:",
        label=f"Initial ({airfoil_name.upper()})",
    )
    plt.plot(x, init_af.lower_coordinates(x)[:, 1], "k:")

    plt.plot(
        x,
        van_af.upper_coordinates(x)[:, 1],
        "gray",
        linestyle="--",
        label="Best Vanilla SPSA",
    )
    plt.plot(x, van_af.lower_coordinates(x)[:, 1], "gray", linestyle="--")

    plt.plot(
        x,
        rl_af.upper_coordinates(x)[:, 1],
        "b-",
        linewidth=2.0,
        label="Best RL-SPSA",
    )
    plt.plot(x, rl_af.lower_coordinates(x)[:, 1], "b-", linewidth=2.0)

    plt.title(f"Optimized Airfoil Shape Comparison ({airfoil_name.upper()})")
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.gca().set_aspect("equal")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
