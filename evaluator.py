"""基于 AeroSandbox 与 NeuralFoil 的 Kulfan (18维) 工业级气动与几何综合评估器"""

from typing import Any, Dict, Optional, Tuple
import aerosandbox as asb
import neuralfoil as nf
import numpy as np


class AerodynamicEvaluator:
    """工业级翼型气动与几何约束评估器。

    涵盖航空工业实践中的核心几何约束：
    1. 最大相对厚度及其弦向站位
    2. 关键结构梁站位（前梁、主梁/油箱、后梁、近后缘）最小厚度
    3. 后缘区厚度单调递减（彻底杜绝鱼尾 fish-tailing 与尾部鼓包）
    4. 前缘无量纲钝度半径（杜绝菜刀尖刃与过钝头）
    5. 后缘闭合楔角（杜绝平行或发散后缘）
    6. 无量纲截面面积（结构抗弯刚度与油箱容积）
    7. 最大弯度约束
    8. 吸力面单峰单调性与前中部抗波纹凸性（消除 CST 高阶多项式振荡/驼峰/凹坑）
    9. 俯仰力矩约束（可选，抑制极端后部加载带来的配平阻力）

    采用分级强约束与主导惩罚机制（Dominant Penalty）：
    - 致命缺陷（表面自交叉、厚度崩塌）前置拦截，绝不调用 NeuralFoil，防止神经网络分布外产生虚假超高 L/D。
    - 物理约束一旦发生微小突破，施加阶跃门槛惩罚（30~50点）+ 平滑梯度引导项，
      确保越界在博弈上永远是巨亏的，杜绝优化器突破约束。
    """

    def __init__(
        self,
        reynolds: float = 1e6,
        alpha: float = 5.0,
        min_thickness: float = 0.08,
        max_thickness: float = 0.20,
        min_thickness_loc: float = 0.20,
        max_thickness_loc: float = 0.45,
        min_spar_thicknesses: Optional[Dict[float, float]] = None,
        min_le_radius: float = 0.008,
        max_le_radius: float = 0.05,
        min_te_angle: float = 3.0,
        min_area: float = 0.050,
        max_camber: float = 0.06,
        enforce_aft_monotonicity: bool = True,
        enforce_smoothness: bool = True,
        min_cm: Optional[float] = -0.15,
        penalty_scale: float = 1.0,
    ):
        self.Re = float(reynolds)
        self.alpha = float(alpha)

        # 厚度及站位约束
        self.min_thickness = float(min_thickness)
        self.max_thickness = float(max_thickness)
        self.min_thickness_loc = float(min_thickness_loc)
        self.max_thickness_loc = float(max_thickness_loc)

        # 结构梁站位厚度约束：若未指定，默认与 min_thickness 按工程比例绑定
        if min_spar_thicknesses is not None:
            self.min_spar_thicknesses = {
                float(k): float(v) for k, v in min_spar_thicknesses.items()
            }
        else:
            self.min_spar_thicknesses = {
                0.20: round(self.min_thickness * 0.60, 4),  # 前梁站位 (如 8% 对应 0.048)
                0.40: round(self.min_thickness * 0.70, 4),  # 主梁/油箱区 (如 8% 对应 0.056)
                0.70: round(self.min_thickness * 0.35, 4),  # 后梁站位 (如 8% 对应 0.028)
                0.90: 0.005,                                # 近后缘站位 (防挤压破损)
            }

        # 前缘与后缘几何
        self.min_le_radius = float(min_le_radius)
        self.max_le_radius = float(max_le_radius)
        self.min_te_angle = float(min_te_angle)

        # 截面面积与最大弯度
        self.min_area = float(min_area)
        self.max_camber = float(max_camber)

        # 表面光滑度与单调性
        self.enforce_aft_monotonicity = bool(enforce_aft_monotonicity)
        self.enforce_smoothness = bool(enforce_smoothness)

        # 俯仰力矩约束
        self.min_cm = float(min_cm) if min_cm is not None else None

        # 全局惩罚系数
        self.penalty_scale = float(penalty_scale)

    def _theta_to_kulfan(self, theta) -> Tuple[Dict[str, Any], asb.KulfanAirfoil]:
        """将 18 维向量拆解为 AeroSandbox 与 NeuralFoil 格式"""
        theta = np.asarray(theta, dtype=np.float64)
        u_w = theta[0:8]
        l_w = theta[8:16]
        le_w = float(theta[16])
        te_t = 0.0025

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

    @staticmethod
    def _dominant_penalty(
        violation: float,
        scale: float,
        base_step: float = 40.0,
        k1: float = 120.0,
        k2: float = 400.0,
    ) -> float:
        """工业绝对主导惩罚函数：

        当约束被违背 (violation > 0) 时：
        - 立即施加 base_step 门槛跃阶惩罚（直接抹平潜在气动收益）；
        - 附加连续的一阶与二阶项，为贝叶斯优化（TuRBO）提供朝向可行域的明确下降梯度。
        """
        if violation <= 0.0:
            return 0.0
        rel = violation / scale
        return base_step + k1 * rel + k2 * (rel ** 2)

    def validate_geometry(
        self, kulfan_af: asb.KulfanAirfoil
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """全套工业几何校验与特征提取。

        返回:
            (is_fatal, fatal_penalty, geom_metrics)
        """
        x_sample = np.linspace(0.001, 0.999, 120)
        try:
            yu = kulfan_af.upper_coordinates(x_sample)[:, 1]
            yl = kulfan_af.lower_coordinates(x_sample)[:, 1]
        except Exception:
            return True, 500.0, {}

        if np.any(np.isnan(yu)) or np.any(np.isnan(yl)):
            return True, 500.0, {}

        thickness_sample = yu - yl
        overlap = yl - yu  # 下表面高过上表面即为交叉
        max_overlap = float(np.max(overlap))
        min_thickness_found = float(np.min(thickness_sample))
        max_thickness_sample = float(np.max(thickness_sample))

        # 致命几何缺陷判断：表面反转交叉或严重塌陷
        if max_overlap > 1e-4 or min_thickness_found < 0.0 or max_thickness_sample < 0.02:
            overlap_sum = float(np.sum(np.maximum(0.0, overlap)))
            fatal_penalty = 100.0 + 5000.0 * (overlap_sum + max(0.0, 0.02 - max_thickness_sample))
            idx_tmax = int(np.argmax(thickness_sample))
            return True, fatal_penalty, {
                "t_max": max_thickness_sample,
                "x_t_max": float(x_sample[idx_tmax]),
                "area": 0.0,
                "le_radius": 0.0,
                "te_angle": 0.0,
                "max_camber": 0.0,
                "spar_thicknesses": {},
                "aft_non_mono": 0.0,
                "crest_non_mono": 0.0,
                "waviness": 0.0,
                "max_overlap": max_overlap,
                "min_thickness_found": min_thickness_found,
                "max_thickness_sample": max_thickness_sample,
            }

        # 正常提取各项几何指标
        idx_tmax = int(np.argmax(thickness_sample))
        x_t_max = float(x_sample[idx_tmax])

        try:
            t_max = float(kulfan_af.max_thickness())
        except Exception:
            t_max = max_thickness_sample

        try:
            area = float(kulfan_af.area())
        except Exception:
            area = float(np.trapz(thickness_sample, x_sample))

        try:
            le_radius = float(kulfan_af.LE_radius())
            if np.isnan(le_radius) or le_radius < 0:
                le_radius = 0.0
        except Exception:
            le_radius = 0.0

        try:
            te_angle = float(kulfan_af.TE_angle())
            if np.isnan(te_angle):
                te_angle = 0.0
        except Exception:
            te_angle = 0.0

        try:
            max_camber = float(kulfan_af.max_camber())
        except Exception:
            max_camber = float(np.max(np.abs(0.5 * (yu + yl))))

        # 站位厚度
        spar_thicknesses = {}
        for xs in self.min_spar_thicknesses.keys():
            try:
                spar_thicknesses[xs] = float(kulfan_af.local_thickness(xs))
            except Exception:
                spar_thicknesses[xs] = float(np.interp(xs, x_sample, thickness_sample))

        # 表面光滑度与单调性检测
        # 1. 后缘区厚度单调递减检测 (从最大厚度位置到后缘，厚度不得反常增加)
        aft_thickness = thickness_sample[idx_tmax:]
        diff_aft = np.diff(aft_thickness)
        aft_non_mono = float(np.sum(np.maximum(0.0, diff_aft)))

        # 2. 上表面单峰单调性检测 (最高点之前单调上升，之后单调下降，无局部凹坑)
        idx_yu_crest = int(np.argmax(yu))
        yu_pre = yu[:idx_yu_crest]
        yu_post = yu[idx_yu_crest:]
        crest_non_mono = 0.0
        if len(yu_pre) > 1:
            crest_non_mono += float(np.sum(np.maximum(0.0, -np.diff(yu_pre))))
        if len(yu_post) > 1:
            crest_non_mono += float(np.sum(np.maximum(0.0, np.diff(yu_post))))

        # 3. 前中部上表面凹陷反弯 (Anti-Waviness)
        # 上表面在吸力面前部 (x: 0.05 ~ 0.65) 应保持凸起 (d2_yu <= 0)，正的二阶差分代表凹陷波纹
        d2_yu = np.diff(yu, 2)
        x_mid_mask = (x_sample[1:-1] >= 0.05) & (x_sample[1:-1] <= 0.65)
        waviness = float(np.sum(np.maximum(0.0, d2_yu[x_mid_mask])))

        geom_metrics = {
            "t_max": t_max,
            "x_t_max": x_t_max,
            "area": area,
            "le_radius": le_radius,
            "te_angle": te_angle,
            "max_camber": max_camber,
            "spar_thicknesses": spar_thicknesses,
            "aft_non_mono": aft_non_mono,
            "crest_non_mono": crest_non_mono,
            "waviness": waviness,
            "max_overlap": max_overlap,
        }
        return False, 0.0, geom_metrics

    def calculate_penalties(
        self, geom_metrics: Dict[str, Any], aero: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, float, Dict[str, float], Dict[str, float]]:
        """计算各项工业约束违背量与对应绝对主导惩罚。

        返回:
            (is_feasible, total_penalty, violations, penalties)
        """
        violations = {}
        penalties = {}

        # 1. 最大厚度下界与上界
        t_max = geom_metrics["t_max"]
        if t_max < self.min_thickness:
            v = self.min_thickness - t_max
            violations["t_max_min"] = v
            penalties["t_max_min"] = self._dominant_penalty(v, scale=0.01)
        elif t_max > self.max_thickness:
            v = t_max - self.max_thickness
            violations["t_max_max"] = v
            penalties["t_max_max"] = self._dominant_penalty(v, scale=0.01)

        # 2. 最大厚度弦向站位
        x_t_max = geom_metrics["x_t_max"]
        if x_t_max < self.min_thickness_loc:
            v = self.min_thickness_loc - x_t_max
            violations["x_t_max_min"] = v
            penalties["x_t_max_min"] = self._dominant_penalty(v, scale=0.05, base_step=30.0)
        elif x_t_max > self.max_thickness_loc:
            v = x_t_max - self.max_thickness_loc
            violations["x_t_max_max"] = v
            penalties["x_t_max_max"] = self._dominant_penalty(v, scale=0.05, base_step=30.0)

        # 3. 关键梁站位厚度
        spar_t = geom_metrics.get("spar_thicknesses", {})
        for xs, min_req in self.min_spar_thicknesses.items():
            actual_t = spar_t.get(xs, 0.0)
            if actual_t < min_req:
                v = min_req - actual_t
                key = f"spar_t_{xs:.2f}"
                violations[key] = v
                penalties[key] = self._dominant_penalty(v, scale=0.005, base_step=25.0)

        # 4. 前缘半径
        le_r = geom_metrics["le_radius"]
        if le_r < self.min_le_radius:
            v = self.min_le_radius - le_r
            violations["le_radius_min"] = v
            penalties["le_radius_min"] = self._dominant_penalty(v, scale=0.002, base_step=35.0)
        elif le_r > self.max_le_radius:
            v = le_r - self.max_le_radius
            violations["le_radius_max"] = v
            penalties["le_radius_max"] = self._dominant_penalty(v, scale=0.01, base_step=20.0)

        # 5. 后缘闭合角
        te_angle = geom_metrics["te_angle"]
        if te_angle < self.min_te_angle:
            v = self.min_te_angle - te_angle
            violations["te_angle"] = v
            penalties["te_angle"] = self._dominant_penalty(v, scale=1.0, base_step=30.0)

        # 6. 截面面积
        area = geom_metrics["area"]
        if area < self.min_area:
            v = self.min_area - area
            violations["area_min"] = v
            penalties["area_min"] = self._dominant_penalty(v, scale=0.008, base_step=30.0)

        # 7. 最大弯度
        camber = geom_metrics["max_camber"]
        if camber > self.max_camber:
            v = camber - self.max_camber
            violations["max_camber"] = v
            penalties["max_camber"] = self._dominant_penalty(v, scale=0.01, base_step=25.0)

        # 8. 后缘单调收缩（杜绝鱼尾）
        if self.enforce_aft_monotonicity:
            aft_non_mono = geom_metrics["aft_non_mono"]
            if aft_non_mono > 1e-4:
                violations["aft_non_mono"] = aft_non_mono
                penalties["aft_non_mono"] = self._dominant_penalty(
                    aft_non_mono, scale=0.003, base_step=35.0
                )

        # 9. 表面波纹与凹坑抑制
        if self.enforce_smoothness:
            smooth_violation = geom_metrics["crest_non_mono"] + geom_metrics["waviness"]
            if smooth_violation > 2e-4:
                violations["smoothness"] = smooth_violation
                penalties["smoothness"] = self._dominant_penalty(
                    smooth_violation, scale=0.002, base_step=25.0
                )

        # 10. 俯仰力矩约束 (CM)
        if aero is not None and self.min_cm is not None:
            cm = aero.get("cm", 0.0)
            if cm < self.min_cm:
                v = self.min_cm - cm
                violations["min_cm"] = v
                penalties["min_cm"] = self._dominant_penalty(v, scale=0.02, base_step=20.0)

        total_penalty = float(sum(penalties.values())) * self.penalty_scale
        is_feasible = (len(violations) == 0)
        return is_feasible, total_penalty, violations, penalties

    def evaluate(self, theta) -> float:
        """核心评估接口：

        先进行工业几何前置筛查，致命缺陷直接早退拦截（不调用 NeuralFoil）；
        通过几何筛查后进入 NeuralFoil 气动计算，并施加绝对主导约束惩罚。

        返回: L/D 升阻比 (扣除所有几何与气动违背惩罚项)
        """
        kulfan_dict, kulfan_af = self._theta_to_kulfan(theta)

        # 1. 工业几何前置校验
        is_fatal, fatal_penalty, geom_metrics = self.validate_geometry(kulfan_af)
        if is_fatal:
            # 致命缺陷立即返回重罚，杜绝神经网络分布外幻觉
            return -100.0 - fatal_penalty

        # 2. NeuralFoil 气动计算
        try:
            aero_nf = nf.get_aero_from_kulfan_parameters(
                kulfan_parameters=kulfan_dict,
                alpha=self.alpha,
                Re=self.Re,
                model_size="xxxlarge",
            )
            cl = float(np.asarray(aero_nf["CL"]).item())
            cd = float(np.asarray(aero_nf["CD"]).item())
            cm = float(np.asarray(aero_nf["CM"]).item())
        except Exception:
            return -50.0

        if cd <= 1e-5 or np.isnan(cl) or np.isnan(cd) or np.isnan(cm):
            return -50.0

        raw_ld = cl / cd
        aero_dict = {"cl": cl, "cd": cd, "cm": cm, "ld": raw_ld}

        # 3. 约束惩罚计算
        is_feasible, total_penalty, _, _ = self.calculate_penalties(
            geom_metrics, aero=aero_dict
        )

        return raw_ld - total_penalty

    def evaluate_detailed(self, theta) -> Dict[str, Any]:
        """获取详细气动参数与全套工业几何校验报告（用于测试输出与分析）"""
        kulfan_dict, kulfan_af = self._theta_to_kulfan(theta)

        is_fatal, fatal_penalty, geom_metrics = self.validate_geometry(kulfan_af)

        if is_fatal:
            return {
                "cl": 0.0,
                "cd": 1.0,
                "cm": 0.0,
                "ld": 0.0,
                "ld_penalized": -100.0 - fatal_penalty,
                "t_max": geom_metrics.get("t_max", 0.0),
                "x_t_max": geom_metrics.get("x_t_max", 0.0),
                "area": geom_metrics.get("area", 0.0),
                "le_radius": geom_metrics.get("le_radius", 0.0),
                "te_angle": geom_metrics.get("te_angle", 0.0),
                "max_camber": geom_metrics.get("max_camber", 0.0),
                "spar_thicknesses": geom_metrics.get("spar_thicknesses", {}),
                "is_feasible": False,
                "is_fatal": True,
                "violations": {"fatal_geometry": fatal_penalty},
                "penalties": {"fatal_geometry": fatal_penalty},
                "total_penalty": fatal_penalty,
                "kulfan_af": kulfan_af,
            }

        cl, cd, cm, raw_ld = 0.0, 1.0, 0.0, 0.0
        try:
            aero_nf = nf.get_aero_from_kulfan_parameters(
                kulfan_parameters=kulfan_dict,
                alpha=self.alpha,
                Re=self.Re,
                model_size="xxxlarge",
            )
            cl = float(np.asarray(aero_nf["CL"]).item())
            cd = float(np.asarray(aero_nf["CD"]).item())
            cm = float(np.asarray(aero_nf["CM"]).item())
            raw_ld = cl / cd if cd > 1e-5 else 0.0
        except Exception:
            cl, cd, cm, raw_ld = 0.0, 1.0, 0.0, 0.0

        aero_dict = {"cl": cl, "cd": cd, "cm": cm, "ld": raw_ld}
        is_feasible, total_penalty, violations, penalties = self.calculate_penalties(
            geom_metrics, aero=aero_dict
        )

        penalized_ld = raw_ld - total_penalty

        return {
            "cl": cl,
            "cd": cd,
            "cm": cm,
            "ld": raw_ld,
            "ld_penalized": penalized_ld,
            "t_max": geom_metrics.get("t_max", 0.0),
            "x_t_max": geom_metrics.get("x_t_max", 0.0),
            "area": geom_metrics.get("area", 0.0),
            "le_radius": geom_metrics.get("le_radius", 0.0),
            "te_angle": geom_metrics.get("te_angle", 0.0),
            "max_camber": geom_metrics.get("max_camber", 0.0),
            "spar_thicknesses": geom_metrics.get("spar_thicknesses", {}),
            "is_feasible": is_feasible and not is_fatal,
            "is_fatal": is_fatal,
            "violations": violations,
            "penalties": penalties,
            "total_penalty": total_penalty,
            "kulfan_af": kulfan_af,
        }

