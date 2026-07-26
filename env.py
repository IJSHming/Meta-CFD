"""基于 Gymnasium 的元优化环境（SPSA 边界截断与 Spall 准则）"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MetaSPSA_Airfoil_Env(gym.Env):
    """
    符合 Gymnasium 规范的元优化环境[cite: 1]。
    RL 智能体 (SAC) 将通过动作空间直接调节 SPSA 的超参数，目标是最大化累积势能差。
    """

    def __init__(self, evaluator, init_theta=None, dim=10, max_cfd_calls=300):
        super().__init__()
        self.evaluator = evaluator
        self.dim = dim
        self.max_cfd_calls = max_cfd_calls

        # 默认初始几何为 NACA 0012 参数
        if init_theta is None:
            self.init_theta = np.array([0.15] * 5 + [-0.15] * 5, dtype=np.float32)
        else:
            self.init_theta = np.array(init_theta, dtype=np.float32)

        # 动作空间：SPSA 学习率与扰动规模的调节乘子 [0.1, 2.0]
        self.action_space = spaces.Box(low=0.1, high=2.0, shape=(2,), dtype=np.float32)
        # 状态空间：[全局归一化最佳值, 近期改善幅度, 预算剩余率, 当前基准a, 当前基准c]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cfd_calls = 0
        self.theta = self.init_theta.copy()

        # 记录当前参数与最佳参数
        self.best_f = self.evaluator.evaluate(self.theta[:5], self.theta[5:])
        self.best_theta = self.theta.copy()
        self.last_improvement = 0.0

        # 套用 Spall 标准参数准则[cite: 1]
        self.a_base = 0.0002
        self.c_base = 0.008
        self.A_const = 10.0

        return self._get_obs(), {}

    def _get_obs(self):
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
        a_multiplier, c_multiplier = action[0], action[1]

        # 计算 SPSA 当前步长
        iter_k = self.cfd_calls / 3.0
        ak = (self.a_base * a_multiplier) / ((iter_k + self.A_const) ** 0.602)
        ck = (self.c_base * c_multiplier) / ((iter_k + 1.0) ** 0.101)

        # SPSA 双向伯努利随机扰动
        delta = np.random.choice([-1, 1], size=self.dim)
        theta_plus = self.theta + ck * delta
        theta_minus = self.theta - ck * delta

        # 对扰动点做严格物理边界截断，防止评估器返回 -50.0 导致虚假梯度[cite: 1]
        theta_plus[:5] = np.clip(theta_plus[:5], 0.05, 0.45)
        theta_plus[5:] = np.clip(theta_plus[5:], -0.45, -0.05)
        theta_minus[:5] = np.clip(theta_minus[:5], 0.05, 0.45)
        theta_minus[5:] = np.clip(theta_minus[5:], -0.45, -0.05)

        f_plus = self.evaluator.evaluate(theta_plus[:5], theta_plus[5:])
        f_minus = self.evaluator.evaluate(theta_minus[:5], theta_minus[5:])
        self.cfd_calls += 2

        # 梯度估计与参数更新
        g = (f_plus - f_minus) / (2 * ck * delta)
        # 限制单步更新最大幅度 (防弹射)
        delta_theta = ak * g
        max_step = 0.02
        delta_theta = np.clip(delta_theta, -max_step, max_step)
        self.theta = self.theta + delta_theta

        # 中心点截断
        self.theta[:5] = np.clip(self.theta[:5], 0.05, 0.45)
        self.theta[5:] = np.clip(self.theta[5:], -0.45, -0.05)

        f_new = self.evaluator.evaluate(self.theta[:5], self.theta[5:])
        self.cfd_calls += 1

        # 奖励机制：基于势能的奖励塑形[cite: 1]
        improvement = max(0, f_new - self.best_f)
        self.last_improvement = improvement
        reward = improvement * 5.0 - 0.1

        if f_new > self.best_f:
            self.best_f = f_new
            self.best_theta = self.theta.copy()

        terminated = False
        truncated = bool(self.cfd_calls >= self.max_cfd_calls)

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            {
                "best_f": self.best_f,
                "f_new": f_new,
                "calls": self.cfd_calls,
                "best_theta": self.best_theta,
            },
        )
