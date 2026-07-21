import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.special import comb
from stable_baselines3 import SAC


# ==========================================
# 模块一：CST 参数化与目标函数(气动)评估器
# ==========================================
class AerodynamicEvaluator:
    """
    轻量化气动评估器：负责 CST 坐标生成与气动性能评估。
    实际工业应用中应在此类内部通过 subprocess 调用 XFOIL 或 OpenFOAM，
    为保证脚本轻量化运行，此处使用数学流形模拟非凸气动特性。
    """

    def __init__(self, reynolds=1e6, mach=0.1, alpha=5.0, base_airfoil="NACA0012"):
        self.Re = reynolds
        self.Mach = mach
        self.alpha = alpha
        self.base = base_airfoil

    def _bernstein(self, n, i, x):
        """计算 Bernstein 多项式基函数"""
        return comb(n, i) * (x**i) * ((1 - x) ** (n - i))

    def generate_cst_coordinates(self, weights_u, weights_l, dz_te=0.0, n_points=100):
        """利用 CST 算法生成翼型上下表面坐标集合"""
        x = np.linspace(0, 1, n_points)
        class_func = np.sqrt(x) * (1.0 - x)  # N1=0.5, N2=1.0 圆头尖尾特征

        n_u = len(weights_u) - 1
        n_l = len(weights_l) - 1
        y_u, y_l = np.zeros_like(x), np.zeros_like(x)

        for i in range(len(weights_u)):
            y_u += weights_u[i] * self._bernstein(n_u, i, x)
        for i in range(len(weights_l)):
            y_l += weights_l[i] * self._bernstein(n_l, i, x)

        y_upper = class_func * y_u + x * (dz_te / 2.0)
        y_lower = class_func * y_l - x * (dz_te / 2.0)
        return x, y_upper, y_lower

    def evaluate(self, weights_u, weights_l):
        """
        返回评估指标，即升阻比 (L/D)。
        此处模拟 XFOIL 行为：遇到无效几何返回严重惩罚。
        """
        # 几何物理约束过滤：上表面权重必须为正，下表面权重必须为负，否则交叉
        if np.any(weights_u < 0) or np.any(weights_l > 0):
            return -50.0  # 非物理构型的重罚

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
        super(MetaSPSA_Airfoil_Env, self).__init__()
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

        self.a_base = 0.5  # SPSA 基础学习率
        self.c_base = 0.1  # SPSA 基础扰动规模

        return self._get_obs(), {}

    def _get_obs(self):
        """提炼元级状态特征"""
        budget_ratio = self.cfd_calls / self.max_cfd_calls
        obs = np.array(
            [
                self.best_f / 100.0,
                self.last_improvement / 10.0,
                budget_ratio,
                self.a_base,
                self.c_base,
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action):
        """
        RL 干预环节：智能体输出动作调节 SPSA，底层 SPSA 消耗 CFD 预算。
        """
        # 解析 RL 的动态调节乘子
        a_multiplier, c_multiplier = action[0], action[1]

        # 计算 SPSA 当前代步长 (引入 RL 乘子)
        ak = (self.a_base * a_multiplier) / ((self.cfd_calls + 10) ** 0.602)
        ck = (self.c_base * c_multiplier) / ((self.cfd_calls + 1) ** 0.101)

        # SPSA 伯努利随机扰动近似梯度
        delta = np.random.choice([-1, 1], size=self.dim)
        theta_plus = self.theta + ck * delta
        theta_minus = self.theta - ck * delta

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

        if f_new > self.best_f:
            self.best_f = f_new

        terminated = False
        truncated = bool(self.cfd_calls >= self.max_cfd_calls)

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            {"best_f": self.best_f, "calls": self.cfd_calls},
        )


# ==========================================
# 模块三：实验调度管道 (训练与泛化验证)
# ==========================================
if __name__ == "__main__":
    # 阶段一与二：在源任务 (NACA 0012, Re=1e6) 上训练 SAC 元智能体
    print(">>> [Phase 1 & 2] 构建源任务环境并启动 SAC 元训练...")
    evaluator_source = AerodynamicEvaluator(
        reynolds=1e6, alpha=5.0, base_airfoil="NACA0012"
    )
    env_train = MetaSPSA_Airfoil_Env(evaluator_source, max_cfd_calls=150)

    # 实例化 SAC 算法，启用最大熵探索机制
    model = SAC(
        "MlpPolicy",
        env_train,
        verbose=0,
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=64,
    )
    model.learn(total_timesteps=5000)
    print(">>> 元智能体训练完成。")

    # 阶段三：同分布源任务评估
    print("\n>>> [Phase 3] 同分布环境 (NACA 0012) RL-SPSA 性能评估...")
    obs, _ = env_train.reset()
    done = False
    while not done:
        action, _states = model.predict(
            obs, deterministic=True
        )  # 使用确定性策略进行推断
        obs, reward, terminated, truncated, info = env_train.step(action)
        done = terminated or truncated
    print(
        f"[验证结果] 最终最佳升阻比: {info['best_f']:.2f}, CFD 调用次数: {info['calls']}"
    )

    # 阶段四：跨流场、跨翼型的零样本泛化测试
    print(
        "\n>>> [Phase 4] 零样本泛化测试 (目标域：NACA 4412类流形, Re=3e6, Alpha=3.0) ..."
    )
    evaluator_target = AerodynamicEvaluator(
        reynolds=3e6, alpha=3.0, base_airfoil="NACA4412"
    )
    env_test = MetaSPSA_Airfoil_Env(evaluator_target, max_cfd_calls=150)

    obs, _ = env_test.reset()
    done = False
    while not done:
        # 直接复用基于源任务训练好的 SAC 模型控制优化器，不进行任何微调
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_test.step(action)
        done = terminated or truncated

    print(
        f"[泛化结果] 最终最佳升阻比: {info['best_f']:.2f}, CFD 调用次数: {info['calls']}"
    )
