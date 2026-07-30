"""基于 Gymnasium 的 Kulfan (18 维) 翼型 SPSA 元优化环境"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MetaSPSA_Airfoil_Env(gym.Env):
    def __init__(self, evaluator, init_theta, dim=18, max_cfd_calls=501):
        super().__init__()
        self.evaluator = evaluator
        self.dim = dim
        self.max_cfd_calls = max_cfd_calls
        self.init_theta = np.array(init_theta, dtype=np.float32)

        # 动作空间：SPSA 学习率与扰动规模调节乘子 [3^-1, 3^1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # 状态空间：[lg(Re)-5, Alpha/10, 总体提升率, 近期改善率, 预算剩余率, 当前a, 当前c]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

    def set_task(self, evaluator, init_theta):
        """设置当前任务工况与初始坐标"""
        self.evaluator = evaluator
        self.init_theta = np.array(init_theta, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cfd_calls = 0
        self.theta = self.init_theta.copy()

        # 初始点评估
        self.f_init = self.evaluator.evaluate(self.theta)
        self.f = self.f_init

        # best 记录仅用于最终返回输出，不参与任何改善率/状态计算
        self.best_f = self.f_init
        self.best_theta = self.theta.copy()
        self.last_improvement = 0.0

        # Spall 标准准则参数
        self.a_base = 0.0002  # 0.0002
        self.c_base = 0.008  # 0.008
        self.A_const = 10.0

        self.ak = self.a_base / (self.A_const**0.602)
        self.ck = self.c_base

        return self._get_obs(), {}

    def _get_obs(self):
        # 雷诺数取 lg 并减 5 (1e5~1e7 映射至 0~2)
        log10_re_norm = float(np.log10(self.evaluator.Re) - 5.0)
        # 攻角除以 10
        alpha_norm = float(self.evaluator.alpha / 10.0)

        budget_ratio = self.cfd_calls / self.max_cfd_calls
        # 基于当前 self.f 计算总体提升率（非 self.best_f）
        rel_total_improvement = (self.f - self.f_init) / (abs(self.f_init) + 1e-6)
        # 允许为负的近期改善率
        rel_recent_improvement = self.last_improvement / (abs(self.f) + 1e-6)
        return np.array(
            [
                log10_re_norm,
                alpha_norm,
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

        # SPSA 步长与扰动量计算
        iter_k = self.cfd_calls / 3.0
        ak = (self.a_base * a_multiplier) / ((iter_k + self.A_const) ** 0.602)
        ck = (self.c_base * c_multiplier) / ((iter_k + 1.0) ** 0.101)
        self.ak, self.ck = ak, ck

        # SPSA 随机伯努利扰动
        delta = np.random.choice([-1, 1], size=self.dim)
        theta_plus = self.theta + ck * delta
        theta_minus = self.theta - ck * delta

        # 去除所有 clip 截断操作，直接评估物理真实值
        f_plus = self.evaluator.evaluate(theta_plus)
        f_minus = self.evaluator.evaluate(theta_minus)
        self.cfd_calls += 2

        # 梯度估计与无 clip 参数更新
        g = (f_plus - f_minus) / (2 * ck * delta)
        self.theta = self.theta + ak * g

        # 中心点评估
        f_new = self.evaluator.evaluate(self.theta)
        self.cfd_calls += 1

        # 允许改善率为负数
        self.last_improvement = f_new - self.f
        self.f = f_new

        # 仅在需要时更新 best 供最终结果输出
        if self.f > self.best_f:
            self.best_f = self.f
            self.best_theta = self.theta.copy()

        # reward 简单定义为当前的总体提升率乘以 1 步
        rel_total_improvement = (self.f - self.f_init) / (abs(self.f_init) + 1e-6)
        reward = rel_total_improvement * 1.0

        terminated = False
        truncated = bool(self.cfd_calls >= self.max_cfd_calls)

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            {
                "best_f": self.best_f,
                "f_new": self.f,
                "calls": self.cfd_calls,
                "best_theta": self.best_theta,
            },
        )


class MultiConditionAirfoilEnv(gym.Env):
    """多工况-多几何组合训练包装器：在 reset 时顺序切换组合，形成统一训练进度条"""

    def __init__(self, condition_grid, max_cfd_calls=501):
        super().__init__()
        self.condition_grid = condition_grid
        self.n_conditions = len(condition_grid)
        self.current_idx = 0

        first_eval, first_theta = self.condition_grid[0]
        self.inner_env = MetaSPSA_Airfoil_Env(
            first_eval, first_theta, max_cfd_calls=max_cfd_calls
        )
        self.action_space = self.inner_env.action_space
        self.observation_space = self.inner_env.observation_space

    def reset(self, seed=None, options=None):
        evaluator, init_theta = self.condition_grid[
            self.current_idx % self.n_conditions
        ]
        self.inner_env.set_task(evaluator, init_theta)
        self.current_idx += 1
        return self.inner_env.reset(seed=seed, options=options)

    def step(self, action):
        return self.inner_env.step(action)
