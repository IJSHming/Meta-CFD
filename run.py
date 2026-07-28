import os

import numpy as np
from stable_baselines3 import SAC

from env import MetaSPSA_Airfoil_Env
from evaluator import AerodynamicEvaluator
from utils import plot_airfoil_comparison, plot_convergence_results, run_evaluation

# ==========================================
# 全局超参数定义
# ==========================================
CFD_BUDGET = 501  # 单次优化评估预算
RE1, AOA1 = 1e6, 5.0  # 源任务工况 (NACA 0012)
RE2, AOA2 = 3e6, 3.0  # 泛化目标工况 (NACA 4412)

# CST 初始特征参数定义
NACA0012_THETA = np.array(
    [0.15, 0.15, 0.15, 0.15, 0.15, -0.15, -0.15, -0.15, -0.15, -0.15]
)
NACA4412_THETA = np.array(
    [0.22, 0.28, 0.20, 0.14, 0.08, -0.08, -0.10, -0.08, -0.05, -0.02]
)


def train_and_save(model_path="sac_metaspsa", timesteps=5000):
    """
    训练阶段：实例化环境与 SAC 模型，训练完成后自动保存到本地磁盘
    """
    print(f">>> [Training] 启动 SAC 元训练 (Timesteps={timesteps})...")
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
    model.learn(total_timesteps=timesteps, progress_bar=True)

    # 保存模型 (SB3 会自动附加 .zip 拓展名)
    model.save(model_path)
    print(f">>> [Saved] 模型已成功保存至: {model_path}.zip")


def load_and_test(model_path="sac_metaspsa"):
    """
    测试阶段：从磁盘加载模型，依次执行 Phase 3 (源任务) 与 Phase 4 (泛化测试) 评估并画图
    """
    model_file = f"{model_path}.zip" if not model_path.endswith(".zip") else model_path
    if not os.path.exists(model_file):
        print(
            f"❌ [Error] 找不到模型文件 '{model_file}'，请先运行 'python main.py train' 进行训练！"
        )
        return

    print(f">>> [Testing] 成功加载本地模型: '{model_file}'")
    model = SAC.load(model_path)

    # 初始化评估器与测试环境
    evaluator_source = AerodynamicEvaluator(
        reynolds=RE1, alpha=AOA1, min_thickness=0.10
    )
    evaluator_target = AerodynamicEvaluator(
        reynolds=RE2, alpha=AOA2, min_thickness=0.10
    )

    # ---------------------------------------------------------
    # 阶段三：同分布源任务评估 (NACA 0012)
    # ---------------------------------------------------------
    print("\n>>> [Phase 3] 评估源任务 (NACA 0012)...")
    env_vanilla_3 = MetaSPSA_Airfoil_Env(
        evaluator_source, init_theta=NACA0012_THETA, max_cfd_calls=CFD_BUDGET
    )
    env_rl_3 = MetaSPSA_Airfoil_Env(
        evaluator_source, init_theta=NACA0012_THETA, max_cfd_calls=CFD_BUDGET
    )

    v_c3, v_f3, info_v3, theta_s_vanilla = run_evaluation(env_vanilla_3, is_rl=False)
    r_c3, r_f3, info_r3, theta_s_rl = run_evaluation(env_rl_3, model=model, is_rl=True)

    print(f"[Phase 3 - 原版 SPSA] 最佳升阻比: {info_v3['best_f']:.2f}")
    print(f"[Phase 3 - RL - SPSA] 最佳升阻比: {info_r3['best_f']:.2f}")

    # ---------------------------------------------------------
    # 阶段四：跨工况 + 跨几何零样本泛化评估 (NACA 4412)
    # ---------------------------------------------------------
    print("\n>>> [Phase 4] 评估零样本泛化任务 (NACA 4412)...")
    env_vanilla_4 = MetaSPSA_Airfoil_Env(
        evaluator_target, init_theta=NACA4412_THETA, max_cfd_calls=CFD_BUDGET
    )
    env_rl_4 = MetaSPSA_Airfoil_Env(
        evaluator_target, init_theta=NACA4412_THETA, max_cfd_calls=CFD_BUDGET
    )

    v_c4, v_f4, info_v4, theta_t_vanilla = run_evaluation(env_vanilla_4, is_rl=False)
    r_c4, r_f4, info_r4, theta_t_rl = run_evaluation(env_rl_4, model=model, is_rl=True)

    print(f"[Phase 4 - 原版 SPSA] 最佳升阻比: {info_v4['best_f']:.2f}")
    print(f"[Phase 4 - RL - SPSA] 最佳升阻比: {info_r4['best_f']:.2f}")

    # ---------------------------------------------------------
    # 绘制对比结果
    # ---------------------------------------------------------
    print("\n>>> [Plotting] 正在绘制评估曲线与衍生翼型对比...")
    plot_convergence_results(
        p3_data=(v_c3, v_f3, r_c3, r_f3), p4_data=(v_c4, v_f4, r_c4, r_f4)
    )

    theta_dict = {
        "s_base": NACA0012_THETA,
        "s_vanilla": theta_s_vanilla,
        "s_rl": theta_s_rl,
        "t_base": NACA4412_THETA,
        "t_vanilla": theta_t_vanilla,
        "t_rl": theta_t_rl,
    }
    plot_airfoil_comparison(evaluator_source, evaluator_target, theta_dict)
