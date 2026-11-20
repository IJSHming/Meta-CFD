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


def _get_canonical_naca0012_theta() -> np.ndarray:
    """获取 NACA 0012 标准 18 维 Kulfan 参数 (与 RL-Airfoil/src/airfoil_utils.py 完全一致)."""
    af = asb.Airfoil("naca0012").to_kulfan_airfoil(n_weights_per_side=8)
    return np.concatenate([
        np.asarray(af.upper_weights, dtype=np.float64),
        np.asarray(af.lower_weights, dtype=np.float64),
        [float(af.leading_edge_weight), float(af.TE_thickness)],
    ])


def get_airfoil_bounds(init_theta: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """返回与 RL-Airfoil 100% 比特级一致的全局统一 18 维 Kulfan 搜索空间 [lb, ub].

    固定同一全局归一化坐标映射 [0, 1]^18 <-> [lb, ub]，确保元训练任务与下游测试任务
    在潜在特征空间 phi(x_unit) 中物理语义完全对齐。
    """
    base_theta = _get_canonical_naca0012_theta()
    lb = np.zeros(18, dtype=np.float64)
    ub = np.zeros(18, dtype=np.float64)

    lb[0:8] = np.maximum(0.01, base_theta[0:8] - 0.20)
    ub[0:8] = base_theta[0:8] + 0.25
    lb[8:16] = base_theta[8:16] - 0.25
    ub[8:16] = np.minimum(0.05, base_theta[8:16] + 0.20)
    lb[16] = -0.05
    ub[16] = 0.05
    lb[17] = 0.0010
    ub[17] = 0.0060
    return lb, ub


def get_airfoil_parameters(airfoil_name: str) -> np.ndarray:
    """提取指定翼型的 18 维 Kulfan (CST) 几何参数并裁剪至统一搜索盒内。"""
    af = asb.Airfoil(airfoil_name).to_kulfan_airfoil(n_weights_per_side=8)
    theta = np.concatenate(
        [
            np.asarray(af.upper_weights, dtype=np.float64),
            np.asarray(af.lower_weights, dtype=np.float64),
            np.array([float(af.leading_edge_weight)], dtype=np.float64),
            np.array([float(af.TE_thickness)], dtype=np.float64),
        ]
    )
    lb, ub = get_airfoil_bounds()
    return np.clip(theta, lb, ub)


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
        "naca4412": {
            "category": "大弯度高升力型 (High-Camber / High-Lift Airfoil)",
            "camber": "4.0% (高弯度)",
            "thickness": "12.0%",
            "application": "STOL 短距起降机、高空长航时无人机 (HALE)、低速升力面",
            "features": "大弯度产生极高零攻角升力系数，在低中攻角下具备高升阻比。",
        },
        "naca2415": {
            "category": "厚结构翼型 (Thick Structural Airfoil)",
            "camber": "2.0% (适度弯度)",
            "thickness": "15.0%",
            "application": "机翼内段、高抗弯结构需求机翼",
            "features": "较大相对厚度提供高抗弯模量与油箱容积，边界层逆压梯度显著。",
        },
        "b737b": {
            "category": "现代喷气客机跨音速/后加载工程型 (Modern Jet Transport / Aft-loaded Profile)",
            "camber": "0.8%",
            "thickness": "12.7%",
            "application": "波音 737 干线客机真实中段机翼剖面",
            "features": "平坦上表面延缓激波，前缘钝度大缓和吸力峰，后部具备典型的超临界后加载修形。",
        },
    }
