import os
import subprocess
import uuid

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from scipy.special import comb
from stable_baselines3 import SAC

CFD = 300
"""CFD调用预算"""
RE1 = 1e6
"""训练雷诺数"""
RE2 = 3e6
"""泛化雷诺数"""
AOA1 = 5.0
"""训练攻角"""
AOA2 = 3.0
"""泛化攻角"""


# ==========================================
# 模块一：CST 参数化与目标函数(气动)评估器
# ==========================================
class AerodynamicEvaluator:
    """
    轻量化气动评估器：负责 CST 坐标生成与气动性能评估。
    实际工业应用中应在此类内部通过 subprocess 调用 XFOIL 或 OpenFOAM，
    为保证脚本轻量化运行，此处使用数学流形模拟非凸气动特性。
    """

    def __init__(
        self,
        reynolds=1e6,
        mach=0.1,
        alpha=5.0,
        base_airfoil="NACA0012",
        xfoil_cmd="xfoil.exe",
    ):
        self.Re = reynolds
        self.Mach = mach
        self.alpha = alpha
        self.base = base_airfoil
        self.xfoil_cmd = xfoil_cmd

    def _bernstein(self, n, i, x):
        """计算 Bernstein 多项式基函数"""
        return comb(n, i) * (x**i) * ((1 - x) ** (n - i))

    def generate_cst_coordinates(self, weights_u, weights_l, dz_te=0.0, n_points=100):
        """利用 CST 算法生成翼型上下表面坐标集合，接入XFoil后才会被调用"""
        x = np.linspace(0, 1, n_points)
        class_func = np.sqrt(x) * (1.0 - x)  # N1=0.5, N2=1.0 圆头尖尾特征
        n_u, n_l = len(weights_u) - 1, len(weights_l) - 1
        y_u, y_l = np.zeros_like(x), np.zeros_like(x)

        for i in range(len(weights_u)):
            y_u += weights_u[i] * self._bernstein(n_u, i, x)
        for i in range(len(weights_l)):
            y_l += weights_l[i] * self._bernstein(n_l, i, x)

        y_u = class_func * y_u + x * (dz_te / 2.0)
        y_l = class_func * y_l - x * (dz_te / 2.0)

        # 拼接为 XFOIL 要求的流向 (顺时针或逆时针闭合)
        x_u, y_u = x[::-1], y_u[::-1]  # 上表面：尾缘 -> 前缘
        x_l, y_l = x[1:], y_l[1:]  # 下表面：前缘 -> 尾缘

        x_coords = np.concatenate([x_u, x_l])
        y_coords = np.concatenate([y_u, y_l])

        return x_coords, y_coords

    def _write_airfoil_file(self, x_coords, y_coords, filepath):
        """将坐标保存为 XFOIL 可读的 .dat 文件"""
        with open(filepath, "w") as f:
            f.write("CST_AIRFOIL\n")
            f.writelines(f" {x:10.6f}  {y:10.6f}\n" for x, y in zip(x_coords, y_coords))

    def _run_xfoil(self, dat_path, polar_path):
        """生成 XFOIL 控制指令脚本并用 subprocess 执行[cite: 1]"""
        # 构建 XFOIL 命令流
        xfoil_commands = f"""
LOAD {dat_path}
PANE
OPER
VISC {self.Re}
MACH {self.Mach}
ITER 150
PACC
{polar_path}

ALFA {self.alpha}
PACC
QUIT
"""
        try:
            # 启动进程执行 XFOIL，设置 5 秒超时保护防死锁
            subprocess.run(
                [self.xfoil_cmd],
                input=xfoil_commands,
                text=True,
                capture_output=True,
                timeout=5.0,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, None

        # 解析导出的 Polar 极曲线文件
        if not os.path.exists(polar_path):
            return None, None

        cl, cd = None, None
        with open(polar_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split()
                # 寻找数据行（通常匹配包含 alpha, CL, CD 的数字行）
                if len(parts) >= 7:
                    try:
                        alpha_val = float(parts[0])
                        if abs(alpha_val - self.alpha) < 1e-2:
                            cl = float(parts[1])
                            cd = float(parts[2])
                            break
                    except ValueError:
                        continue
        return cl, cd

    def evaluate(self, weights_u, weights_l):
        """
        返回评估指标，即升阻比 (L/D)。最佳值在85和100上下。
        此处模拟 XFOIL 行为：遇到无效几何返回严重惩罚。
        """
        # 几何物理约束过滤：上表面权重必须为正，下表面权重必须为负，否则交叉
        if np.any(weights_u < 0) or np.any(weights_l > 0):
            return -50.0  # 非物理构型的重罚

        if "xfoil" in self.xfoil_cmd:
            x_coords, y_coords = self.generate_cst_coordinates(weights_u, weights_l)

            # 2. 为防止多线程/频繁调用产生文件冲突，生成唯一的临时文件名
            uid = str(uuid.uuid4())[:8]
            dat_path = f"temp_airfoil_{uid}.dat"
            polar_path = f"temp_polar_{uid}.txt"

            try:
                self._write_airfoil_file(x_coords, y_coords, dat_path)
                cl, cd = self._run_xfoil(dat_path, polar_path)

                # 3. 如果 XFOIL 边界层分离不收敛或计算出非物理的 cd<=0，返还高额惩罚值[cite: 1]
                if cl is None or cd is None or cd <= 1e-5:
                    return -50.0

                ld_ratio = cl / cd
                return float(ld_ratio)

            finally:
                # 4. 无论成功与否，立即清理磁盘上的临时文件
                if os.path.exists(dat_path):
                    os.remove(dat_path)
                if os.path.exists(polar_path):
                    os.remove(polar_path)

        # 根据目标工况设定一个"虚拟的最优权重中心"，模拟优化地形
        if self.base == "NACA0012":
            opt_u = np.array([0.15, 0.20, 0.15, 0.10, 0.05])
            opt_l = np.array([-0.15, -0.20, -0.15, -0.10, -0.05])
            base_ld = 75.0
        else:  # NACA4412 或其他泛化目标
            opt_u = np.array([0.18, 0.25, 0.20, 0.15, 0.08])
            opt_l = np.array([-0.10, -0.12, -0.10, -0.05, -0.02])
            base_ld = 90.0

        # 计算当前参数空间与流形最优解的距离
        dist_u = np.linalg.norm(weights_u - opt_u)
        dist_l = np.linalg.norm(weights_l - opt_l)

        # 引入高度非线性与流场噪音 (模拟早期失速或附面层转捩敏感性)
        noise = np.random.normal(0, 1.0)
        ld_ratio = (
            base_ld
            - 200.0 * (dist_u**2 + dist_l**2)
            + 10.0 * np.cos(10 * dist_u)
            + noise
        )

        return float(ld_ratio)


# ==========================================
# 模块二：基于 SPSA 的元优化强化学习环境
# ==========================================
class MetaSPSA_Airfoil_Env(gym.Env):
    """
    符合 Gymnasium 规范的元优化环境。
    RL 智能体 (SAC) 将通过动作空间直接调节 SPSA 的超参数，目标是最大化累积势能差。
    """

    def __init__(self, evaluator, dim=10, max_cfd_calls=200):
        super().__init__()
        self.evaluator = evaluator
        self.dim = dim
        self.max_cfd_calls = max_cfd_calls

        # 动作空间：SPSA 学习率与扰动规模的调节乘子 [0.1, 2.0]
        self.action_space = spaces.Box(low=0.1, high=2.0, shape=(2,), dtype=np.float32)

        # 状态空间：[全局归一化最佳值, 近期改善幅度, 预算剩余率, 当前基准a, 当前基准c]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cfd_calls = 0

        # 初始化起点设计变量 (NACA 0012 的大致 CST 权重)
        self.theta = np.array([0.11] * 5 + [-0.11] * 5)
        self.best_f = self.evaluator.evaluate(self.theta[:5], self.theta[5:])
        self.last_improvement = 0.0

        # 套用 Spall 标准参数准则
        self.a_base = 0.0002  # 调试过程：0.5 -> 0.005
        self.c_base = 0.008  # 调试过程：0.1 -> 0.01
        self.A_const = 10.0  # 稳定衰减常数

        return self._get_obs(), {}

    def _get_obs(self):
        """提炼元级状态特征"""
        budget_ratio = self.cfd_calls / self.max_cfd_calls
        return np.array(
            [
                self.best_f / 100.0,
                self.last_improvement / 10.0,
                budget_ratio,
                self.a_base,
                self.c_base,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        """
        RL 干预环节：智能体输出动作调节 SPSA，底层 SPSA 消耗 CFD 预算。
        """
        # 解析 RL 的动态调节乘子
        a_multiplier, c_multiplier = action[0], action[1]

        # 计算 SPSA 当前代步长 (引入 RL 乘子)
        iter_k = self.cfd_calls / 3.0
        ak = (self.a_base * a_multiplier) / ((iter_k + self.A_const) ** 0.602)
        ck = (self.c_base * c_multiplier) / ((iter_k + 1.0) ** 0.101)

        # SPSA 伯努利随机扰动近似梯度
        delta = np.random.choice([-1, 1], size=self.dim)
        theta_plus = self.theta + ck * delta
        theta_minus = self.theta - ck * delta

        # [修复核心] 必须对扰动点进行物理边界截断，防止评估器返回 -50.0 的虚假梯度
        theta_plus[:5] = np.clip(theta_plus[:5], 0.01, 0.4)
        theta_plus[5:] = np.clip(theta_plus[5:], -0.4, -0.01)
        theta_minus[:5] = np.clip(theta_minus[:5], 0.01, 0.4)
        theta_minus[5:] = np.clip(theta_minus[5:], -0.4, -0.01)

        f_plus = self.evaluator.evaluate(theta_plus[:5], theta_plus[5:])
        f_minus = self.evaluator.evaluate(theta_minus[:5], theta_minus[5:])
        self.cfd_calls += 2

        # 更新设计参数 (目标是最大化)
        g = (f_plus - f_minus) / (2 * ck * delta)
        self.theta = self.theta + ak * g

        # 边界约束截断 (防止参数爆炸导致形状离谱)
        self.theta[:5] = np.clip(self.theta[:5], 0.01, 0.4)
        self.theta[5:] = np.clip(self.theta[5:], -0.4, -0.01)

        # 评估中心更新点
        f_new = self.evaluator.evaluate(self.theta[:5], self.theta[5:])
        self.cfd_calls += 1

        # ======= 奖励机制：基于势能的奖励塑形 =======
        improvement = max(0, f_new - self.best_f)
        self.last_improvement = improvement

        # 密集奖励：正向改善给予重奖，无改善则施加轻微惩罚以迫使压缩预算
        reward = improvement * 5.0 - 0.2

        self.best_f = max(self.best_f, f_new)

        terminated = False
        truncated = bool(self.cfd_calls >= self.max_cfd_calls)

        # 新增 f_new 输出，反映当前步的真实气动性能
        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            {"best_f": self.best_f, "f_new": f_new, "calls": self.cfd_calls},
        )


# ==========================================
# 模块三：实验调度管道（训练与泛化验证）与可视化对比
# ==========================================
def run_evaluation(env, model=None, is_rl=False):
    """辅助推断运行函数，用于收集绘图所需的真实 f_new 轨迹"""
    obs, _ = env.reset()
    done = False
    calls_history = [0]
    fnew_history = [env.best_f]  # 初始中心点的性能

    while not done:
        if is_rl and model is not None:
            # 使用 SAC 元智能体输出动态超参数乘子，使用确定性策略进行推断
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.array([1.0, 1.0], dtype=np.float32)  # 原版SPSA固定乘子

        obs, _, terminated, truncated, info = env.step(action)
        calls_history.append(info["calls"])
        fnew_history.append(info["f_new"])  # 追踪真实 f_new
        done = terminated or truncated

    return calls_history, fnew_history, info


if __name__ == "__main__":
    # ---------------------------------------------------------
    # 阶段一与二：在源任务 (NACA 0012, Re=1e6) 上训练 SAC 元智能体
    # ---------------------------------------------------------
    print(">>> [Phase 1 & 2] 构建源任务环境并启动 SAC 元训练...")
    evaluator_source = AerodynamicEvaluator(
        reynolds=RE1, alpha=AOA1, base_airfoil="NACA0012"
    )
    env_train = MetaSPSA_Airfoil_Env(evaluator_source, max_cfd_calls=CFD)

    # 实例化 SAC 算法，启用最大熵探索机制，progress_bar=True 显示训练进度条
    model = SAC(
        "MlpPolicy",
        env_train,
        verbose=0,
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=64,
    )
    model.learn(total_timesteps=5000, progress_bar=True)
    print(">>> 元智能体训练完成。")

    # ---------------------------------------------------------
    # 阶段三：同分布环境 (NACA 0012) 真实收敛轨迹对比
    # ---------------------------------------------------------
    print("\n>>> [Phase 3] 同分布环境 (NACA 0012) 评估...")
    # 1. 运行原版 SPSA 并记录数据
    env_vanilla_3 = MetaSPSA_Airfoil_Env(evaluator_source, max_cfd_calls=CFD)

    vanilla_calls_3, vanilla_f_3, info_v3 = run_evaluation(env_vanilla_3, is_rl=False)
    print(f"[Phase 3 - 原版SPSA] 最终最佳升阻比: {info_v3['best_f']:.2f}")

    rl_calls_3, rl_f_3, info_r3 = run_evaluation(env_train, model=model, is_rl=True)
    print(f"[Phase 3 - RL-SPSA] 最终最佳升阻比: {info_r3['best_f']:.2f}")

    # ---------------------------------------------------------
    # 阶段四：跨工况零样本泛化测试 (NACA 4412) 真实收敛轨迹对比
    # ---------------------------------------------------------
    print("\n>>> [Phase 4] 零样本泛化测试 (NACA 4412) 评估...")
    evaluator_target = AerodynamicEvaluator(
        reynolds=RE2, alpha=AOA2, base_airfoil="NACA4412"
    )

    env_vanilla_4 = MetaSPSA_Airfoil_Env(evaluator_target, max_cfd_calls=CFD)
    vanilla_calls_4, vanilla_f_4, info_v4 = run_evaluation(env_vanilla_4, is_rl=False)
    print(f"[Phase 4 - 原版SPSA] 最终最佳升阻比: {info_v4['best_f']:.2f}")

    # 直接复用基于源任务训练好的 SAC 模型控制优化器，不进行任何微调
    env_rl_4 = MetaSPSA_Airfoil_Env(evaluator_target, max_cfd_calls=CFD)
    rl_calls_4, rl_f_4, info_r4 = run_evaluation(env_rl_4, model=model, is_rl=True)
    print(f"[Phase 4 - RL-SPSA] 最终最佳升阻比: {info_r4['best_f']:.2f}")

    # ---------------------------------------------------------
    # 绘制双子图对比
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 子图1：阶段三源任务
    axes[0].plot(
        vanilla_calls_3,
        vanilla_f_3,
        label="Vanilla SPSA (f_new)",
        color="gray",
        alpha=0.7,
        linestyle="--",
    )
    axes[0].plot(
        rl_calls_3, rl_f_3, label="RL-SPSA (f_new)", color="blue", linewidth=1.5
    )
    axes[0].set_title("Phase 3: Source Task (NACA 0012) Convergence")
    axes[0].set_xlabel("CFD Call Budget")
    axes[0].set_ylabel("Current Objective Value (L/D)")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.7)

    # 子图2：阶段四泛化任务
    axes[1].plot(
        vanilla_calls_4,
        vanilla_f_4,
        label="Vanilla SPSA (f_new)",
        color="gray",
        alpha=0.7,
        linestyle="--",
    )
    axes[1].plot(
        rl_calls_4,
        rl_f_4,
        label="RL-SPSA (Zero-shot f_new)",
        color="green",
        linewidth=1.5,
    )
    axes[1].set_title("Phase 4: Generalization Task (NACA 4412) Convergence")
    axes[1].set_xlabel("CFD Call Budget")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    plt.show()
