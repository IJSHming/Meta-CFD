"""MALIBO 训练集生成系统全局配置文件与参数规范。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence
import numpy as np


# 默认基础路径
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DEFAULT_DATA_DIR = PACKAGE_DIR / "data"
DEFAULT_TASKS_DIR = DEFAULT_DATA_DIR / "tasks"
DEFAULT_MANIFEST_PATH = DEFAULT_DATA_DIR / "meta_dataset_manifest.json"
DEFAULT_PT_PATH = DEFAULT_DATA_DIR / "malibo_train_dataset.pt"

# 5 种迥异的起始基准翼型
BASELINE_AIRFOILS: Sequence[str] = (
    "naca0012",  # 1. 纯对称翼型 (零弯度，反转对称)
    "naca2412",  # 2. 经典常规民航/通航巡航型 (适度弯度，平缓失速)
    "naca6412",  # 3. 大弯度高升力翼型 (深弯度，极高 CL0)
    "naca2418",  # 4. 大厚度强结构/高油箱容积型 (18% 厚度，近上限)
    "b737b",     # 5. 真实现代喷气式客机超临界/后加载工程型 (平坦吸力面与后弯)
)

# 10 级雷诺数分布 (1e6 到 1e7，分为 10 个点: 1M ~ 10M)
DEFAULT_REYNOLDS_GRID: Sequence[float] = (
    1.0e6,
    2.0e6,
    3.0e6,
    4.0e6,
    5.0e6,
    6.0e6,
    7.0e6,
    8.0e6,
    9.0e6,
    1.0e7,
)

# 11 级攻角分布 (0 到 10 度，分为 11 个点: 0, 1, 2, ..., 10 度)
DEFAULT_ALPHA_GRID: Sequence[float] = (
     0.0,
     1.0,
     2.0,
     3.0,
     4.0,
     5.0,
     6.0,
     7.0,
     8.0,
     9.0,
    10.0,
)

# 紧凑版 5 级攻角分布 (聚焦核心升阻比区域)
COMPACT_ALPHA_GRID: Sequence[float] = (
     0.0,
     2.5,
     5.0,
     7.5,
    10.0,
)

# 极速测试配置 (用于流水线连通性冒烟测试)
QUICK_AIRFOILS: Sequence[str] = ("naca0012", "naca2412")
QUICK_REYNOLDS: Sequence[float] = (1.0e6, 2.0e6)
QUICK_ALPHAS: Sequence[float] = (2.0, 5.0)


@dataclass
class OptimizationConfig:
    """CMA-ES 优化超参数配置类 (严格适配 18 维气动设计最佳实践)"""

    max_evals: int = 500             # 严格单任务 500 步评估上限
    dim: int = 18                    # Kulfan 参数维度
    sigma0: float = 0.20             # 归一化空间初始步长 (覆盖约 40% 有效范围)
    popsize: int = 12                # 种群规模 (CMA-ES 最佳经验值: 4 + floor(3 * ln(18)) = 12)
    bounds: tuple = (0.0, 1.0)       # 坐标超矩形上下界
    seed: int = 42                   # 基础随机数种子


@dataclass
class MALIBOTrainingConfig:
    """MALIBO 神经网络训练集格式规范配置"""

    gamma: float = 1.0 / 3.0         # 论文标准优选分位数比例 (tau = Phi^-1(gamma))
    minimization: bool = True        # 目标值转换为最小化 y = -penalized_ld
    feature_dim: int = 64            # ResFFN 隐层与特征维度
    task_embed_dim: int = 50         # 任务嵌入向量维度
