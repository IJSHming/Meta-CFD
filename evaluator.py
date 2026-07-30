"""基于 AeroSandbox 与 NeuralFoil 的 Kulfan (18维) 气动评估器"""

import aerosandbox as asb
import neuralfoil as nf
import numpy as np


class AerodynamicEvaluator:
    def __init__(self, reynolds=1e6, alpha=5.0, min_thickness=0.08, max_thickness=0.22):
        self.Re = float(reynolds)
        self.alpha = float(alpha)
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness

    def _theta_to_kulfan(self, theta):
        """将 18 维向量拆解为 AeroSandbox 与 NeuralFoil 格式"""
        theta = np.asarray(theta, dtype=np.float64)
        u_w = theta[0:8]
        l_w = theta[8:16]
        le_w = float(theta[16])
        # 保证后缘厚度不为负值
        te_t = max(0.0, float(theta[17]))

        kulfan_dict = {
            "upper_weights": u_w,
            "lower_weights": l_w,
            "leading_edge_weight": le_w,
            "TE_thickness": te_t,
        }

        kulfan_af = asb.KulfanAirfoil(
            upper_weights=u_w,
            lower_weights=l_w,
            leading_edge_weight=le_w,
            TE_thickness=te_t,
        )
        return kulfan_dict, kulfan_af

    def evaluate(self, theta):
        """核心评估接口：直接传入 Kulfan parameters 进入 NeuralFoil。

        返回: (ld_ratio, is_valid)
        """
        kulfan_dict, kulfan_af = self._theta_to_kulfan(theta)

        # 厚度约束校验
        t_max = kulfan_af.max_thickness()
        thickness_penalty = 0.0
        if t_max < self.min_thickness:
            thickness_penalty = 500.0 * (self.min_thickness - t_max) ** 2 + 20.0 * (
                self.min_thickness - t_max
            )
        elif t_max > self.max_thickness:
            thickness_penalty = 500.0 * (t_max - self.max_thickness) ** 2 + 20.0 * (
                t_max - self.max_thickness
            )

        # 上下表面交叉性物理校验
        x = np.linspace(0.001, 0.999, 100)
        y_u = kulfan_af.upper_coordinates(x)[:, 1]
        y_l = kulfan_af.lower_coordinates(x)[:, 1]
        overlap = y_l - y_u
        cross_penalty = 0.0
        if np.any(overlap > 0):
            cross_penalty = 1000.0 * float(np.sum(overlap[overlap > 0]))

        # NeuralFoil 气动计算 (直接传入 Kulfan parameters)
        aero = nf.get_aero_from_kulfan_parameters(
            kulfan_parameters=kulfan_dict,
            alpha=self.alpha,
            Re=self.Re,
            model_size="medium",
        )
        cl = float(np.asarray(aero["CL"]).item())
        cd = float(np.asarray(aero["CD"]).item())

        if cd <= 1e-5 or np.isnan(cl) or np.isnan(cd):
            return -50.0

        return (cl / cd) - thickness_penalty - cross_penalty

    def evaluate_detailed(self, theta):
        """获取详细参数（供测试模式 print 输出与可视化绘图）"""
        kulfan_dict, kulfan_af = self._theta_to_kulfan(theta)
        aero = nf.get_aero_from_kulfan_parameters(
            kulfan_parameters=kulfan_dict,
            alpha=self.alpha,
            Re=self.Re,
            model_size="xxxlarge",
        )
        cl = float(np.asarray(aero["CL"]).item())
        cd = float(np.asarray(aero["CD"]).item())
        ld = cl / cd if cd > 1e-5 else 0.0
        t_max = float(kulfan_af.max_thickness())

        return {
            "cl": cl,
            "cd": cd,
            "ld": ld,
            "t_max": t_max,
            "kulfan_af": kulfan_af,
        }
