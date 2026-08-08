"""基于 Gymnasium 与 pycma 的 Kulfan (18 维) 翼型 CMA-ES 元优化环境"""

import cma
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MetaCMAES_Airfoil_Env(gym.Env):
    def __init__(self, evaluator, init_theta, dim=18, max_cfd_calls=500):
        super().__init__()
        self.evaluator = evaluator
        self.dim = dim
        self.max_cfd_calls = max_cfd_calls
        self.init_theta = np.array(init_theta, dtype=np.float32)

        # 动作空间 (2维 continuous [-1, 1]):
        # action[0]: 步长 sigma 缩放系数 (映射为 2.0^action[0])
        # action[1]: 种群规模 lambda (映射至 6 ~ 24 的整数)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 状态空间 (10维):
        # [lg(Re)-5, Alpha/10, 总体提升率, 近期改善率, 预算剩余率,
        #  lg10(cond(C)), 路径范数比, 当前sigma, 种群适应度标准差, 停滞计数率]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

    def set_task(self, evaluator, init_theta):
        """设置当前任务工况与初始几何坐标"""
        self.evaluator = evaluator
        self.init_theta = np.array(init_theta, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cfd_calls = 0
        self.theta = self.init_theta.copy()

        # 初始点评估
        self.f_init = self.evaluator.evaluate(self.theta)
        self.cfd_calls += 1
        self.f = self.f_init

        self.best_f = self.f_init
        self.best_theta = self.theta.copy()
        self.last_fitnesses = [self.f_init]
        self.stagnation_count = 0

        # 初始化 CMA-ES 实例
        # 设初始步长为 0.1 (适应归一化后的 CST 参数)
        self.sigma0 = 0.01
        self.es = cma.CMAEvolutionStrategy(
            self.theta,
            self.sigma0,
            {"popsize": 12, "verbose": -9, "seed": seed if seed is not None else 0},
        )

        # 记录前两代中心点与位移，用于计算方向一致性与位移速度
        self.prev_mean = self.es.mean.copy()
        self.prev_delta_mean = np.zeros(self.dim, dtype=np.float32)

        return self._get_obs(), {}

    def _get_obs(self):
        # 1. 物理工况与历史状态
        log10_re_norm = float(np.log10(self.evaluator.Re) - 5.0)
        alpha_norm = float(self.evaluator.alpha / 10.0)
        budget_ratio = float(self.cfd_calls / self.max_cfd_calls)
        rel_total_improvement = float(
            (self.best_f - self.f_init) / (abs(self.f_init) + 1e-6)
        )
        rel_recent_improvement = float((self.f - self.f_init) / (abs(self.f) + 1e-6))

        # 2. CMA-ES 内部动力学特征
        cond_C = np.linalg.cond(self.es.C)  # pyright: ignore[reportArgumentType]
        log10_cond_C = float(np.log10(np.clip(cond_C, 1.0, 1e8)))

        # 进化路径范数比 ||p_sigma|| / expected_norm

        # 特征 A: 连续两代移动方向的余弦相似度 (范围 [-1, 1])
        # >0 表示顺风直追，<0 表示反弹震荡，完全替代原 norm_ps 逻辑
        curr_delta = self.es.mean - self.prev_mean
        norm_curr = np.linalg.norm(curr_delta)
        norm_prev = np.linalg.norm(self.prev_delta_mean)

        if norm_curr > 1e-8 and norm_prev > 1e-8:
            directional_coherence = float(
                np.dot(curr_delta, self.prev_delta_mean) / (norm_curr * norm_prev)
            )
        else:
            directional_coherence = 0.0

        # 特征 B: 当前代中心的归一化推进速度 ||Δm|| / σ
        curr_sigma = float(self.es.sigma)
        step_velocity = float(norm_curr / (curr_sigma + 1e-8))

        fit_std = float(
            np.std(self.last_fitnesses) / (abs(np.mean(self.last_fitnesses)) + 1e-6)
        )
        stag_ratio = float(self.stagnation_count / 20.0)

        return np.array(
            [
                log10_re_norm,
                alpha_norm,
                rel_total_improvement,
                rel_recent_improvement,
                budget_ratio,
                log10_cond_C,
                directional_coherence,
                step_velocity,
                fit_std,
                stag_ratio,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        # 1. 计算当代瞬时调节因子 gamma_t (例如 [-1, 1] 映射至 [0.5, 2.0])
        gamma_t = float(2.0 ** action[0])

        # 2. 临时施加给 CMA-ES 用于当代采样
        self.es.sigma *= gamma_t

        # 3. 动态种群规模 lambda (6 ~ 24)
        popsize = int(12 * 2 ** action[1])

        # 4. CMA-ES 采样与评估
        solutions = self.es.ask(number=popsize)
        fitnesses = [self.evaluator.evaluate(sol) for sol in solutions]
        self.cfd_calls += len(solutions)
        self.last_fitnesses = fitnesses

        # 注意: NeuralFoil 返回升阻比 L/D (最大化目标)，而 pycma 默认求极小值，故取负号
        # 5. CMA-ES 矩阵与步长原生更新 (tell 内部会自动完成原生 CSA 演化)
        self.es.tell(solutions, [-fit for fit in fitnesses])

        # 6. 除以 gamma_t，剥离 RL 的累积效应，保留原生 CMA-ES 演化基准！
        self.es.sigma /= gamma_t

        # --- D. 性能跟踪与更新 ---
        current_best_in_pop = max(fitnesses)
        f_new = float(np.mean(fitnesses))

        if current_best_in_pop > self.best_f:
            self.best_f = current_best_in_pop
            self.best_theta = solutions[np.argmax(fitnesses)].copy()
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

        self.f = f_new

        # --- E. Reward 设定 ---
        # 极小化 CFD 调用 + 极大化适应度提升
        rel_total_improvement = (self.best_f - self.f_init) / (abs(self.f_init) + 1e-6)
        step_improvement = max(0.0, current_best_in_pop - self.f_init)
        # 每次评估扣除小额 CFD 调用惩罚，激励算法用更少的评估步数突破性能
        reward = float(
            rel_total_improvement * 1.0 + step_improvement * 2.0 - 0.01 * len(solutions)
        )

        terminated = False
        truncated = bool(self.cfd_calls >= self.max_cfd_calls)

        # 在 step() 的 return 前更新历史中心点与位移向量
        self.prev_delta_mean = (self.es.mean - self.prev_mean).copy()
        self.prev_mean = self.es.mean.copy()

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
    """多工况-多几何组合训练包装器"""

    def __init__(self, condition_grid, max_cfd_calls=500):
        super().__init__()
        self.condition_grid = condition_grid
        self.n_conditions = len(condition_grid)
        self.current_idx = 0

        first_eval, first_theta = self.condition_grid[0]
        self.inner_env = MetaCMAES_Airfoil_Env(
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
