"""主运行入口（阶段 1~4 调度）"""

import random

import numpy as np
import torch
from stable_baselines3 import SAC

from env import MetaSPSA_Airfoil_Env
from evaluator import AerodynamicEvaluator
from utils import plot_airfoil_comparison, plot_convergence_results, run_evaluation

# ==========================================
# 实验超参数设定
# ==========================================
CFD_BUDGET = 666  # 单次优化评估预算
TRAIN_STEPS = CFD_BUDGET // 3 * 30  # SAC 元训练 Timesteps

RE1, AOA1 = 1e6, 5.0  # 源任务工况 (NACA 0012)
RE2, AOA2 = 3e6, 3.0  # 泛化目标工况 (NACA 4412)

# CST 初始特征参数定义
NACA0012_THETA = np.array(
    [0.15, 0.15, 0.15, 0.15, 0.15, -0.15, -0.15, -0.15, -0.15, -0.15]
)
NACA4412_THETA = np.array(
    [0.22, 0.28, 0.20, 0.14, 0.08, -0.08, -0.10, -0.08, -0.05, -0.02]
)


def set_global_seeds(seed=42):
    """锁定所有随机种子，保证代码 100% 可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    # ---------------------------------------------------------
    # 阶段一与二：在源任务 (NACA 0012) 上训练 SAC 元智能体
    # ---------------------------------------------------------
    print(">>> [Phase 1 & 2] 构建源任务环境并启动 SAC 元训练...")
    evaluator_source = AerodynamicEvaluator(
        reynolds=RE1, alpha=AOA1, min_thickness=0.10
    )
    env_train = MetaSPSA_Airfoil_Env(
        evaluator_source, init_theta=NACA0012_THETA, max_cfd_calls=CFD_BUDGET
    )

    model = SAC(
        "MlpPolicy",
        env_train,
        verbose=0,
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=64,
    )
    model.learn(total_timesteps=TRAIN_STEPS, progress_bar=True)
    print(">>> 元智能体训练完成。")

    set_global_seeds()  # 传入固定种子号

    # ---------------------------------------------------------
    # 阶段三：同分布源任务测试 (NACA 0012)
    # ---------------------------------------------------------
    print("\n>>> [Phase 3] 源任务 (NACA 0012) 对比评估...")
    env_vanilla_3 = MetaSPSA_Airfoil_Env(
        evaluator_source, init_theta=NACA0012_THETA, max_cfd_calls=CFD_BUDGET
    )
    v_c3, v_f3, info_v3, theta_s_vanilla = run_evaluation(env_vanilla_3, is_rl=False)

    r_c3, r_f3, info_r3, theta_s_rl = run_evaluation(env_train, model=model, is_rl=True)

    print(f"[Phase 3 - 原版 SPSA] 最佳升阻比: {info_v3['best_f']:.2f}")
    print(f"[Phase 3 - RL - SPSA] 最佳升阻比: {info_r3['best_f']:.2f}")

    # ---------------------------------------------------------
    # 阶段四：跨工况 + 跨几何零样本泛化测试 (NACA 4412)
    # ---------------------------------------------------------
    print("\n>>> [Phase 4] 零样本泛化测试 (NACA 4412) 对比评估...")
    evaluator_target = AerodynamicEvaluator(
        reynolds=RE2, alpha=AOA2, min_thickness=0.10
    )

    env_vanilla_4 = MetaSPSA_Airfoil_Env(
        evaluator_target, init_theta=NACA4412_THETA, max_cfd_calls=CFD_BUDGET
    )
    v_c4, v_f4, info_v4, theta_t_vanilla = run_evaluation(env_vanilla_4, is_rl=False)

    env_rl_4 = MetaSPSA_Airfoil_Env(
        evaluator_target, init_theta=NACA4412_THETA, max_cfd_calls=CFD_BUDGET
    )
    r_c4, r_f4, info_r4, theta_t_rl = run_evaluation(env_rl_4, model=model, is_rl=True)

    print(f"[Phase 4 - 原版 SPSA] 最佳升阻比: {info_v4['best_f']:.2f}")
    print(f"[Phase 4 - RL - SPSA] 最佳升阻比: {info_r4['best_f']:.2f}")

    # ---------------------------------------------------------
    # 绘制结果图表
    # ---------------------------------------------------------
    # 1. 收敛曲线对比图
    plot_convergence_results(
        p3_data=(v_c3, v_f3, r_c3, r_f3), p4_data=(v_c4, v_f4, r_c4, r_f4)
    )

    # 2. 最终翼型几何与气动数据对比图 (包含 Baseline, Vanilla, RL 共 4 种衍生翼型)
    theta_dict = {
        "s_base": NACA0012_THETA,
        "s_vanilla": theta_s_vanilla,
        "s_rl": theta_s_rl,
        "t_base": NACA4412_THETA,
        "t_vanilla": theta_t_vanilla,
        "t_rl": theta_t_rl,
    }
    plot_airfoil_comparison(evaluator_source, evaluator_target, theta_dict)
