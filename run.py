"""训练与测试流程控制逻辑"""

import os

import numpy as np
from stable_baselines3 import SAC

from env import MetaSPSA_Airfoil_Env, MultiConditionAirfoilEnv
from evaluator import AerodynamicEvaluator
from utils import (
    build_training_grid,
    get_kulfan_parameters_from_name,
    plot_airfoil_shapes_and_print_metrics,
    plot_test_convergence,
    run_evaluation,
)

CFD_BUDGET = 501  # 每轮评估预算 (对应 501 / 3 = 167 个 step)


def train_and_save(model_path="sac_metaspsa", rounds_per_combo=1):
    """训练阶段：遍历 10*10*5=500 种组合，每种组合训练 rounds_per_combo 轮。"""
    print(">>> 正在生成 10*10*5 (500种) 工况与几何组合网格...")
    grid = build_training_grid()

    total_episodes = len(grid) * rounds_per_combo
    steps_per_episode = CFD_BUDGET // 3
    total_timesteps = total_episodes * steps_per_episode

    print(">>> [Training] SAC 元训练参数配置:")
    print(f"    - 每工况组合训练轮数 n: {rounds_per_combo}")
    print(f"    - 总 Episode 轮数: {total_episodes}")
    print(f"    - 每轮 Step 数: {steps_per_episode}")
    print(f"    - 最终总 Timesteps: {total_timesteps}")

    train_env = MultiConditionAirfoilEnv(grid, max_cfd_calls=CFD_BUDGET)

    # 针对 8.35 万步的 SAC 推荐超参数配置
    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=0,
        learning_rate=3e-4,  # 平稳收敛的学习率
        buffer_size=100000,  # 完整容纳所有历史样本，防止早崩
        batch_size=256,  # 提高高维连续空间下梯度的估计稳定性
        learning_starts=1000,  # 前期探索步数
    )

    # 单一进度条展示整个训练过程
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save(model_path)
    print(f">>> [Saved] 模型已成功保存至: {model_path}.zip")


def load_and_test(model_path, airfoil_name, reynolds, alpha):
    """测试阶段：根据命令行强指定的起始翼型、雷诺数与攻角跑 10 次测试，

    画真实平均收敛曲线，并输出/绘制各自 10 次中的最佳结果。
    """
    model_file = f"{model_path}.zip" if not model_path.endswith(".zip") else model_path
    if not os.path.exists(model_file):
        print(f"❌ [Error] 找不到模型文件 '{model_file}'，请先运行训练！")
        return

    print(f">>> [Testing] 加载模型 '{model_file}'...")
    model = SAC.load(model_path)

    evaluator = AerodynamicEvaluator(reynolds=reynolds, alpha=alpha)
    init_theta = get_kulfan_parameters_from_name(airfoil_name)
    test_env = MetaSPSA_Airfoil_Env(evaluator, init_theta, max_cfd_calls=CFD_BUDGET)

    n_runs = 10
    print(
        f">>> 在工况 Re={reynolds:.2e}, Alpha={alpha}°, Airfoil={airfoil_name.upper()} 下执行 {n_runs} 次独立对比实验..."
    )

    vanilla_calls_histories = []
    vanilla_f_histories = []
    vanilla_best_f = -np.inf
    vanilla_best_theta = None

    rl_calls_histories = []
    rl_f_histories = []
    rl_best_f = -np.inf
    rl_best_theta = None

    for _ in range(n_runs):
        # 1. 运行原版 SPSA
        c_v, f_v, b_f_v, b_t_v = run_evaluation(test_env, is_rl=False)
        vanilla_calls_histories.append(c_v)
        vanilla_f_histories.append(f_v)
        if b_f_v > vanilla_best_f:
            vanilla_best_f = b_f_v
            vanilla_best_theta = b_t_v

        # 2. 运行 RL-SPSA
        c_r, f_r, b_f_r, b_t_r = run_evaluation(test_env, model=model, is_rl=True)
        rl_calls_histories.append(c_r)
        rl_f_histories.append(f_r)
        if b_f_r > rl_best_f:
            rl_best_f = b_f_r
            rl_best_theta = b_t_r

    # 计算 10 次运行真实轨迹与呼叫次数的对齐均值
    min_len_v = min(len(h) for h in vanilla_f_histories)
    min_len_r = min(len(h) for h in rl_f_histories)

    vanilla_calls_mean = np.mean(
        [c[:min_len_v] for c in vanilla_calls_histories], axis=0
    )
    vanilla_f_means = np.mean([f[:min_len_v] for f in vanilla_f_histories], axis=0)

    rl_calls_mean = np.mean([c[:min_len_r] for c in rl_calls_histories], axis=0)
    rl_f_means = np.mean([f[:min_len_r] for f in rl_f_histories], axis=0)

    # 1. 绘制收敛曲线
    plot_test_convergence(
        vanilla_calls_mean, vanilla_f_means, rl_calls_mean, rl_f_means
    )

    # 2. 格式化 print 控制台输出指标，并绘制 3 种翼型几何形状
    plot_airfoil_shapes_and_print_metrics(
        evaluator, init_theta, vanilla_best_theta, rl_best_theta, airfoil_name
    )
