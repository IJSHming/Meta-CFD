"""CST 几何生成与 NeuralFoil 气动评估（带物理厚度约束）"""

import neuralfoil as nf
import numpy as np
from scipy.special import comb


class AerodynamicEvaluator:
    """
    基于 NeuralFoil 的轻量化气动评估器：
    - 支持物理相对厚度约束 t_max
    - 纯内存计算，支持 Python 3.12+
    """

    def __init__(self, reynolds=1e6, alpha=5.0, min_thickness=0.10, max_thickness=0.20):
        self.Re = reynolds
        self.alpha = alpha
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness

    def _bernstein(self, n, i, x):
        return comb(n, i) * (x**i) * ((1.0 - x) ** (n - i))

    def generate_cst_coordinates(self, weights_u, weights_l, dz_te=0.0, n_points=100):
        """利用 CST 算法生成翼型上下表面坐标集合及最大相对厚度"""
        x = np.linspace(0, 1, n_points)
        class_func = np.sqrt(x) * (1.0 - x)  # N1=0.5, N2=1.0 圆头尖尾特征

        n_u, n_l = len(weights_u) - 1, len(weights_l) - 1
        y_u_raw, y_l_raw = np.zeros_like(x), np.zeros_like(x)

        for i in range(len(weights_u)):
            y_u_raw += weights_u[i] * self._bernstein(n_u, i, x)
        for i in range(len(weights_l)):
            y_l_raw += weights_l[i] * self._bernstein(n_l, i, x)

        y_upper = class_func * y_u_raw + x * (dz_te / 2.0)
        y_lower = class_func * y_l_raw - x * (dz_te / 2.0)

        # 计算最大相对厚度 t_max
        t_max = float(np.max(y_upper - y_lower))

        return x, y_upper, y_lower, t_max

    def _get_neuralfoil_coords(self, x, y_upper, y_lower):
        """将 CST 坐标拼装为 NeuralFoil 要求的 (N, 2) 逆时针点阵"""
        x_u, y_u = x[::-1], y_upper[::-1]
        x_l, y_l = x[1:], y_lower[1:]
        coords = np.column_stack(
            [np.concatenate([x_u, x_l]), np.concatenate([y_u, y_l])]
        )
        return coords

    def evaluate(self, weights_u, weights_l):
        """核心评估接口：直接返回升阻比 (L/D)"""
        # 1. 物理交叉检查
        if np.any(weights_u < 0) or np.any(weights_l > 0):
            return -50.0

        x, y_upper, y_lower, t_max = self.generate_cst_coordinates(weights_u, weights_l)

        # 2. 几何结构厚度约束
        # 2. 连续软惩罚 (Soft Penalty)，保持梯度平滑连续，杜绝死锁陷阱
        thickness_penalty = 0.0
        if t_max < self.min_thickness:
            thickness_penalty = 500.0 * (self.min_thickness - t_max) ** 2 + 20.0 * (
                self.min_thickness - t_max
            )
        elif t_max > self.max_thickness:
            thickness_penalty = 500.0 * (t_max - self.max_thickness) ** 2 + 20.0 * (
                t_max - self.max_thickness
            )

        coords = self._get_neuralfoil_coords(x, y_upper, y_lower)

        aero = nf.get_aero_from_coordinates(
            coordinates=coords,
            alpha=self.alpha,
            Re=self.Re,
            model_size="medium",
        )
        cl = float(np.asarray(aero["CL"]).item())
        cd = float(np.asarray(aero["CD"]).item())

        if cd <= 1e-5 or np.isnan(cl) or np.isnan(cd):
            return -50.0

        return cl / cd - thickness_penalty  # 平滑扣除惩罚

    def evaluate_detailed(self, weights_u, weights_l):
        """供可视化使用的详细评估接口，返回全套气动参数与几何坐标[cite: 1]"""
        x, y_upper, y_lower, t_max = self.generate_cst_coordinates(weights_u, weights_l)
        coords = self._get_neuralfoil_coords(x, y_upper, y_lower)

        aero = nf.get_aero_from_coordinates(
            coordinates=coords,
            alpha=self.alpha,
            Re=self.Re,
            model_size="xxxlarge",
        )
        cl = float(np.asarray(aero["CL"]).item())
        cd = float(np.asarray(aero["CD"]).item())
        ld = cl / cd if cd > 1e-5 else -50.0

        return {
            "cl": cl,
            "cd": cd,
            "ld": ld,
            "t_max": t_max,
            "x": x,
            "y_upper": y_upper,
            "y_lower": y_lower,
        }
