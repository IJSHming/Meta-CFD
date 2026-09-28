"""5 种迥异起始基准翼型提取、边界构建与几何健康度校验模块。"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import aerosandbox as asb
import numpy as np

# 导入根目录评估器与工具
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluator import AerodynamicEvaluator
from malibo_dataset_generator.config import BASELINE_AIRFOILS


def get_airfoil_parameters(airfoil_name: str) -> np.ndarray:
    """提取指定翼型的 18 维 Kulfan (CST) 几何参数。

    统一将后缘钝化厚度 TE_thickness 对齐为 0.0025，与工业评估器完全契合。
    """
    af = asb.KulfanAirfoil(airfoil_name)
    theta = np.concatenate(
        [
            np.array(af.upper_weights, dtype=np.float64),
            np.array(af.lower_weights, dtype=np.float64),
            np.array([af.leading_edge_weight], dtype=np.float64),
            np.array([max(0.0025, float(af.TE_thickness))], dtype=np.float64),
        ]
    )
    return theta


def get_airfoil_bounds(init_theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """生成 18 维搜索空间的自适应上下界 [lb, ub]。

    严格确保 init_theta 在 [lb, ub] 内，且保证充足且物理合理的探索空间。
    """
    init_theta = np.asarray(init_theta, dtype=np.float64)

    # 默认航空工程经验边界
    default_lb = np.array([-0.2] * 8 + [-0.6] * 8 + [0.0, 0.0025], dtype=np.float64)
    default_ub = np.array([0.6] * 8 + [0.2] * 8 + [0.5, 0.03], dtype=np.float64)

    lb = np.minimum(init_theta - 0.2, default_lb)
    ub = np.maximum(init_theta + 0.2, default_ub)

    # 后缘物理厚度硬约束下限
    lb[17] = 0.0025
    ub[17] = max(float(ub[17]), 0.03)

    # 前缘修形权重大于等于0
    lb[16] = max(0.0, float(lb[16]))

    # 确保严格严格 init_theta 在 [lb, ub] 内部
    lb = np.minimum(lb, init_theta)
    ub = np.maximum(ub, init_theta)

    # 杜绝任何退化区间
    delta = np.maximum(ub - lb, 1e-4)
    ub = np.where(ub <= lb, lb + delta, ub)

    return lb, ub


def verify_airfoil_geometry(
    airfoil_name: str,
    evaluator: Optional[AerodynamicEvaluator] = None,
) -> Dict[str, any]:
    """校验指定翼型在评估器下的初始几何约束与气动健康度。"""
    if evaluator is None:
        evaluator = AerodynamicEvaluator(reynolds=1.0e6, alpha=4.0)

    theta = get_airfoil_parameters(airfoil_name)
    lb, ub = get_airfoil_bounds(theta)
    report = evaluator.evaluate_detailed(theta)

    report["airfoil_name"] = airfoil_name
    report["theta"] = theta
    report["lb"] = lb
    report["ub"] = ub
    report["bounds_valid"] = bool(np.all(lb <= theta) and np.all(theta <= ub))

    return report


def get_baseline_descriptions() -> Dict[str, Dict[str, str]]:
    """获取 5 种起始基准翼型的工程物理背景描述。"""
    return {
        "naca0012": {
            "category": "纯对称翼型 (Symmetric Airfoil)",
            "camber": "0.0% (零弯度)",
            "thickness": "12.0%",
            "application": "直升机旋翼、导弹/火箭气动舵面、常规机翼平尾垂尾",
            "features": "上下剖面完全镜面对称，零攻角零升力，正负攻角流动对称。",
        },
        "naca2412": {
            "category": "经典通航巡航型 (Conventional GA Airfoil)",
            "camber": "1.9% (适度中弯度)",
            "thickness": "12.0%",
            "application": "Cessna 172、经典轻型民用飞机机翼",
            "features": "适度前弯度，平缓失速，在中低雷诺数下拥有极宽的低阻巡航桶。",
        },
        "naca6412": {
            "category": "大弯度高升力型 (High-Camber / High-Lift Airfoil)",
            "camber": "5.8% (大弯度)",
            "thickness": "12.0%",
            "application": "STOL 短距起降机、高空长航时无人机 (HALE)、低速风力发电机",
            "features": "大弯度产生极高的零攻角升力系数 (CL0 ~ 0.7)，适合低速大升力工况。",
        },
        "naca2418": {
            "category": "大厚度强结构/高油箱容积型 (Thick Structural / Root Airfoil)",
            "camber": "1.8%",
            "thickness": "18.0% (接近 20% 约束上限)",
            "application": "大型客机内侧翼根段 (Wing Root)、重载运输机、大型风电叶根",
            "features": "抗弯截面模量大、油箱容积大，但背风逆压梯度极陡，易过早诱导分离。",
        },
        "b737b": {
            "category": "现代喷气客机跨音速/后加载工程型 (Modern Jet Transport / Aft-loaded Profile)",
            "camber": "0.8%",
            "thickness": "12.7%",
            "application": "波音 737 干线客机真实中段机翼剖面",
            "features": "平坦上表面延缓激波，前缘钝度大缓和吸力峰，后部具备典型的超临界后加载修形。",
        },
    }
