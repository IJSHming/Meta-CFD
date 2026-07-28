"""辅助评估函数、收敛曲线绘制与 4 种翼型几何/性能对比绘图"""

import matplotlib.pyplot as plt
import numpy as np


def run_evaluation(env, model=None, is_rl=False):
    """辅助推断运行函数，记录真实 f_new 轨迹并提取最佳设计参数[cite: 1]"""
    obs, _ = env.reset()
    done = False
    calls_history = [0]
    fnew_history = [env.best_f]

    best_theta = env.best_theta.copy()

    while not done:
        if is_rl and model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.array([0.0, 0.0], dtype=np.float32)

        obs, _, terminated, truncated, info = env.step(action)
        calls_history.append(info["calls"])
        fnew_history.append(info["f_new"])
        best_theta = info["best_theta"]
        done = terminated or truncated

    return calls_history, fnew_history, info, best_theta


def moving_average(data, window_size=5):
    """滑动平均以展示平滑趋势线"""
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def plot_convergence_results(p3_data, p4_data):
    """绘制阶段三与阶段四的收敛对比曲线[cite: 1]"""
    _fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    w = 5

    # 阶段三
    v_c3, v_f3, r_c3, r_f3 = p3_data
    axes[0].plot(v_c3, v_f3, color="gray", alpha=0.25)
    axes[0].plot(r_c3, r_f3, color="blue", alpha=0.25)
    axes[0].plot(
        v_c3[w - 1 :],
        moving_average(v_f3, w),
        label="Vanilla SPSA (Trend)",
        color="gray",
        linestyle="--",
        linewidth=2,
    )
    axes[0].plot(
        r_c3[w - 1 :],
        moving_average(r_f3, w),
        label="RL-SPSA (SAC Trend)",
        color="blue",
        linewidth=2.5,
    )
    axes[0].set_title("Phase 3: Source Task (NACA 0012) Convergence")
    axes[0].set_xlabel("CFD Call Budget")
    axes[0].set_ylabel("Current Objective Value (L/D)")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.7)

    # 阶段四
    v_c4, v_f4, r_c4, r_f4 = p4_data
    axes[1].plot(v_c4, v_f4, color="gray", alpha=0.25)
    axes[1].plot(r_c4, r_f4, color="green", alpha=0.25)
    axes[1].plot(
        v_c4[w - 1 :],
        moving_average(v_f4, w),
        label="Vanilla SPSA (Trend)",
        color="gray",
        linestyle="--",
        linewidth=2,
    )
    axes[1].plot(
        r_c4[w - 1 :],
        moving_average(r_f4, w),
        label="RL-SPSA (Zero-shot Trend)",
        color="green",
        linewidth=2.5,
    )
    axes[1].set_title("Phase 4: Generalization Task (NACA 4412) Convergence")
    axes[1].set_xlabel("CFD Call Budget")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    plt.show()


def plot_airfoil_comparison(eval_source, eval_target, theta_dict):
    """
    计算并对比 4 种衍生翼型与原始翼型的几何外形及最终气动性能[cite: 1]。
    包含：Source-Vanilla, Source-RL, Target-Vanilla, Target-RL
    """
    # 提取各翼型的详细评估数据
    res_s_base = eval_source.evaluate_detailed(
        theta_dict["s_base"][:5], theta_dict["s_base"][5:]
    )
    res_s_van = eval_source.evaluate_detailed(
        theta_dict["s_vanilla"][:5], theta_dict["s_vanilla"][5:]
    )
    res_s_rl = eval_source.evaluate_detailed(
        theta_dict["s_rl"][:5], theta_dict["s_rl"][5:]
    )

    res_t_base = eval_target.evaluate_detailed(
        theta_dict["t_base"][:5], theta_dict["t_base"][5:]
    )
    res_t_van = eval_target.evaluate_detailed(
        theta_dict["t_vanilla"][:5], theta_dict["t_vanilla"][5:]
    )
    res_t_rl = eval_target.evaluate_detailed(
        theta_dict["t_rl"][:5], theta_dict["t_rl"][5:]
    )

    _fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 子图 1：源任务翼型比较 (NACA 0012 工况)
    axes[0].plot(
        res_s_base["x"], res_s_base["y_upper"], "k:", label="Baseline (NACA 0012)"
    )
    axes[0].plot(res_s_base["x"], res_s_base["y_lower"], "k:")
    axes[0].plot(
        res_s_van["x"],
        res_s_van["y_upper"],
        "gray",
        linestyle="--",
        label="Vanilla SPSA",
    )
    axes[0].plot(res_s_van["x"], res_s_van["y_lower"], "gray", linestyle="--")
    axes[0].plot(
        res_s_rl["x"], res_s_rl["y_upper"], "b-", linewidth=2, label="RL-SPSA (SAC)"
    )
    axes[0].plot(res_s_rl["x"], res_s_rl["y_lower"], "b-", linewidth=2)
    axes[0].set_title(
        "Source Task (NACA 0012 Init) - Optimized Geometry & Performance Comparison"
    )
    axes[0].set_aspect("equal")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right")

    # 源任务性能文本标注
    txt_s = (
        f"[Baseline] L/D: {res_s_base['ld']:.2f} | CL: {res_s_base['cl']:.3f} | CD: {res_s_base['cd']:.4f} | t_max: {res_s_base['t_max'] * 100:.1f}%\n"
        f"[Vanilla ] L/D: {res_s_van['ld']:.2f} | CL: {res_s_van['cl']:.3f} | CD: {res_s_van['cd']:.4f} | t_max: {res_s_van['t_max'] * 100:.1f}%\n"
        f"[RL-SPSA ] L/D: {res_s_rl['ld']:.2f} | CL: {res_s_rl['cl']:.3f} | CD: {res_s_rl['cd']:.4f} | t_max: {res_s_rl['t_max'] * 100:.1f}%"
    )
    axes[0].text(
        0.02,
        0.05,
        txt_s,
        transform=axes[0].transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.8},
    )

    # 子图 2：泛化任务翼型比较 (NACA 4412 工况)
    axes[1].plot(
        res_t_base["x"], res_t_base["y_upper"], "k:", label="Baseline (NACA 4412)"
    )
    axes[1].plot(res_t_base["x"], res_t_base["y_lower"], "k:")
    axes[1].plot(
        res_t_van["x"],
        res_t_van["y_upper"],
        "gray",
        linestyle="--",
        label="Vanilla SPSA",
    )
    axes[1].plot(res_t_van["x"], res_t_van["y_lower"], "gray", linestyle="--")
    axes[1].plot(
        res_t_rl["x"],
        res_t_rl["y_upper"],
        "g-",
        linewidth=2,
        label="RL-SPSA (Zero-shot)",
    )
    axes[1].plot(res_t_rl["x"], res_t_rl["y_lower"], "g-", linewidth=2)
    axes[1].set_title(
        "Generalization Task (NACA 4412 Init) - Optimized Geometry & Performance Comparison"
    )
    axes[1].set_aspect("equal")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right")

    # 泛化任务性能文本标注
    txt_t = (
        f"[Baseline] L/D: {res_t_base['ld']:.2f} | CL: {res_t_base['cl']:.3f} | CD: {res_t_base['cd']:.4f} | t_max: {res_t_base['t_max'] * 100:.1f}%\n"
        f"[Vanilla ] L/D: {res_t_van['ld']:.2f} | CL: {res_t_van['cl']:.3f} | CD: {res_t_van['cd']:.4f} | t_max: {res_t_van['t_max'] * 100:.1f}%\n"
        f"[RL-SPSA ] L/D: {res_t_rl['ld']:.2f} | CL: {res_t_rl['cl']:.3f} | CD: {res_t_rl['cd']:.4f} | t_max: {res_t_rl['t_max'] * 100:.1f}%"
    )
    axes[1].text(
        0.02,
        0.05,
        txt_t,
        transform=axes[1].transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.show()
