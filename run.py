"""训练与测试流程控制逻辑"""

import os

import numpy as np
from stable_baselines3 import SAC

from env import MetaCMAES_Airfoil_Env, MultiConditionAirfoilEnv
from evaluator import AerodynamicEvaluator
from utils import (
    build_training_grid,
    get_kulfan_parameters_from_name,
    plot_airfoil_shapes_and_print_metrics,
    plot_test_convergence,
    run_evaluation,
)

CFD_BUDGET = 500  # 评估预算


def train_and_save(model_path="sac_metacmaes", rounds_per_combo=1):
    """训练阶段：遍历 500 种工况组合训练 SAC 模型"""
    print(">>> 正在生成 10*10*5 (500种) 工况与几何组合网格...")
    grid = build_training_grid()

    total_episodes = len(grid) * rounds_per_combo
    # CMA-ES 平均代数约为 CFD_BUDGET / lambda_avg (约 40 步)
    steps_per_episode = 40
    total_timesteps = total_episodes * steps_per_episode

    print(">>> [Training] SAC + CMA-ES 元训练参数配置:")
    print(f"    - 每工况组合训练轮数 n: {rounds_per_combo}")
    print(f"    - 总 Episode 轮数: {total_episodes}")
    print(f"    - 预估 Timesteps: {total_timesteps}")

    train_env = MultiConditionAirfoilEnv(grid, max_cfd_calls=CFD_BUDGET)

    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=0,
        learning_rate=3e-4,
        buffer_size=50_000,
        batch_size=256,
        learning_starts=500,
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save(model_path)
    print(f">>> [Saved] 模型已成功保存至: {model_path}.zip")


def load_and_test(model_path, airfoil_name, reynolds, alpha):
    """测试阶段：执行 10 次对比实验"""
    model_file = f"{model_path}.zip" if not model_path.endswith(".zip") else model_path
    if not os.path.exists(model_file):
        print(f"❌ [Error] 找不到模型文件 '{model_file}'，请先运行训练！")
        return

    print(f">>> [Testing] 加载模型 '{model_file}'...")
    model = SAC.load(model_path)

    evaluator = AerodynamicEvaluator(reynolds=reynolds, alpha=alpha)
    init_theta = get_kulfan_parameters_from_name(airfoil_name)
    test_env = MetaCMAES_Airfoil_Env(evaluator, init_theta, max_cfd_calls=CFD_BUDGET)

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
        # 1. 原版 CMA-ES
        c_v, f_v, b_f_v, b_t_v = run_evaluation(test_env, is_rl=False)
        vanilla_calls_histories.append(c_v)
        vanilla_f_histories.append(f_v)
        if b_f_v > vanilla_best_f:
            vanilla_best_f = b_f_v
            vanilla_best_theta = b_t_v

        # 2. RL-CMA-ES
        c_r, f_r, b_f_r, b_t_r = run_evaluation(test_env, model=model, is_rl=True)
        rl_calls_histories.append(c_r)
        rl_f_histories.append(f_r)
        if b_f_r > rl_best_f:
            rl_best_f = b_f_r
            rl_best_theta = b_t_r

    # 计算 10 次运行对齐后的均值
    min_len_v = min(len(h) for h in vanilla_f_histories)
    min_len_r = min(len(h) for h in rl_f_histories)

    vanilla_calls_mean = np.mean(
        [c[:min_len_v] for c in vanilla_calls_histories], axis=0
    )
    vanilla_f_means = np.mean([f[:min_len_v] for f in vanilla_f_histories], axis=0)

    rl_calls_mean = np.mean([c[:min_len_r] for c in rl_calls_histories], axis=0)
    rl_f_means = np.mean([f[:min_len_r] for f in rl_f_histories], axis=0)

    plot_test_convergence(
        vanilla_calls_mean, vanilla_f_means, rl_calls_mean, rl_f_means
    )
    plot_airfoil_shapes_and_print_metrics(
        evaluator, init_theta, vanilla_best_theta, rl_best_theta, airfoil_name
    )
