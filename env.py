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

        # 动作空间：SPSA 学习率与扰动规模的调节乘子 [0.3, 3.0]，用3的-1和+1次方表示
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # 状态空间：[最佳提升率, 近期改善率, 预算剩余率, 当前a, 当前c]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cfd_calls = 0
        self.theta = self.init_theta.copy()

        # 记录当前参数与最佳参数
        self.best_f = self.evaluator.evaluate(self.theta[:5], self.theta[5:])
        self.f_init = self.best_f
        self.f = self.f_init
        self.best_theta = self.theta.copy()
        self.last_improvement = 0.0

        # 套用 Spall 标准参数准则[cite: 1]
        self.a_base = 0.0002
        self.c_base = 0.008
        self.A_const = 10.0

        # 动态维护当前真实步长 ak 与扰动量 ck（初始化第 0 步状态）
        self.ak = self.a_base / (self.A_const**0.602)
        self.ck = self.c_base

        return self._get_obs(), {}

    def _get_obs(self):
        budget_ratio = self.cfd_calls / self.max_cfd_calls
        # 相对初始改善率
        rel_total_improvement = (self.f - self.f_init) / (abs(self.f_init) + 1e-6)
        # 相对近期改善幅度
        rel_recent_improvement = self.last_improvement / (abs(self.f) + 1e-6)
        return np.array(
            [
                rel_total_improvement,
                rel_recent_improvement,
                budget_ratio,
                self.ak,
                self.ck,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        a_multiplier, c_multiplier = 3.0 ** action[0], 3.0 ** action[1]

        # 计算 SPSA 当前步长
        iter_k = self.cfd_calls / 3.0
        ak = (self.a_base * a_multiplier) / ((iter_k + self.A_const) ** 0.602)
        ck = (self.c_base * c_multiplier) / ((iter_k + 1.0) ** 0.101)
        self.ak, self.ck = ak, ck

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
        delta_theta = ak * g
        # 限制单步更新最大幅度 (防弹射)
        # max_step = 0.02
        # delta_theta = np.clip(delta_theta, -max_step, max_step)
        self.theta = self.theta + delta_theta

        # 中心点截断
        self.theta[:5] = np.clip(self.theta[:5], 0.05, 0.45)
        self.theta[5:] = np.clip(self.theta[5:], -0.45, -0.05)

        f_new = self.evaluator.evaluate(self.theta[:5], self.theta[5:])
        self.cfd_calls += 1

        self.last_improvement = max(0, f_new - self.f)

        if f_new > self.best_f:
            self.best_f = f_new
            self.best_theta = self.theta.copy()
        # else:
        #     self.theta = self.best_theta.copy()

        # AUC (曲线下面积) 累积势能奖励
        # 计算当前值相比初始值的相对提升比率
        rel_auc_gain = (f_new - self.f_init) / (abs(self.f_init) + 1e-6)
        # 实时 AUC 奖励：早抬高曲线，后续所有步数都能享受这个高 rel_auc_gain
        reward = rel_auc_gain * 1.0

        terminated = False
        truncated = bool(self.cfd_calls >= self.max_cfd_calls)

        self.f = f_new

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
