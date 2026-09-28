"""MALIBO 架构、元训练、任务自适应与优化超参数配置。

严格对齐 MALIBO 论文 (Pan et al., 2024, Appendix G) 的各项官方默认超参。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ResFFNConfig:
    """特征提取残差前馈网络 (ResFFN) 结构超参 (Appendix G)"""

    in_dim: int = 18                 # 输入参数维度 (如 Kulfan 18 维几何参数)
    hidden_dim: int = 64             # 隐层神经元数量 (4 层，每层 64 单元)
    num_hidden_layers: int = 4       # 残差块/隐层数量
    feature_dim: int = 50            # 潜在特征空间维度 d (论文固定为 50)
    activation: str = "elu"          # 激活函数 (ELU，对齐 Tiao et al. 2021)


@dataclass
class TrainingConfig:
    """元训练 (Meta-Training) 超参数配置 (Appendix G)"""

    lr: float = 1e-3                 # Adam 优化器初始学习率
    lr_decay_factor: float = 0.999   # 学习率指数衰减因子 (每 Epoch 乘以 0.999)
    batch_size: int = 256            # 批大小 B = 256
    epochs: int = 2048               # 最大训练轮数 2,048
    early_stopping_patience: int = 80 # 早停容忍 Epoch 数
    reg_lambda: float = 0.1          # 任务空间高斯先验正则化权重 lambda = 0.1 (Eq. 3)
    gamma: float = 1.0 / 3.0         # LFBO 优质样本分位数比例 (Eq. 2, tau = Phi^-1(gamma))
    seed: int = 42                   # 随机种子
    device: str = "cuda"             # 默认训练硬件 ("cuda" 或 "cpu")
    fused_adam: bool = True          # 是否使用 Fused Adam 优化器 (CUDA 原生融合算子)
    use_compile: bool = True         # 是否使用 torch.compile(mode='reduce-overhead') 固化计算图


@dataclass
class AdaptationConfig:
    """目标任务快速自适应 (Task Adaptation) 超参数 (Appendix G)"""

    lr: float = 1.0                  # L-BFGS 步长
    max_iter: int = 20               # 单步优化最大 L-BFGS 迭代次数
    tolerance_grad: float = 1e-7     # 一阶梯度容差
    tolerance_change: float = 1e-9   # 参数与函数值变动容差
    history_size: int = 100          # L-BFGS 历史窗口大小
    line_search_fn: str = "strong_wolfe" # 线搜索方法


@dataclass
class BoosterConfig:
    """残差梯度提升树 (Gradient Boosting Residual) 配置 (Appendix G)"""

    n_estimators: int = 100          # 最大基学习器树数量
    learning_rate: float = 0.1       # 提升学习率
    min_samples_split: int = 2       # 节点拆分最小样本数
    min_samples_leaf: int = 2        # 叶节点最小样本数 (小样本抗过拟合)
    max_depth: int = 3               # 弱学习器最大深度
    val_ratio: float = 0.3           # 用于早停评估的最佳树数量验证集比例 (30%)
    initial_thompson_steps: int = 20 # 启用残差树模型前纯用 Thompson 采样的过渡步数 (与 n_init 对齐)


@dataclass
class OptimizationConfig:
    """贝叶斯元优化运行配置"""

    n_candidates: int = 5120         # 每一代采集函数极大化采样候选点数 (论文取 5,120)
    budget: int = 200                # 测试优化步数上限
    seed: int = 42                   # 优化器随机种子
    n_init: int = 15                 # 初始采样点数 (基准点 + 空间探索，保障 Hessian/ARD 满秩)
    tr_len_init: float = 0.4         # 初始信任域宽度 (0.4 兼顾全局探索与翼型几何连续性)
    tr_len_restart: float = 0.35     # 信任域收缩至下界后的平滑重启宽度 (杜绝 100x 剧烈跳跃破坏几何)
    tr_len_min: float = 0.5**7       # 约 0.0078125，保障毫米级极精细微调下界
    tr_len_max: float = 1.0          # 最大信任域宽度
    success_tolerance: int = 3       # 连续成功扩张阈值
    failure_tolerance: int = 5       # 连续失败收缩阈值 (适配 200~300 步的敏捷收缩)
    prob_perturb: float = 0.4        # 坐标微扰概率 (防止 18 维全维跳变破坏翼型几何)
    gamma_init: float = 1.0 / 3.0    # 初始优质样本分位数比例
    gamma_min: float = 0.06          # 动态退火分位数下限 (锁定前 6% 极值前沿)
    exploration_floor: float = 0.15  # 拉普拉斯 Thompson 采样探索方差地板 (防止高步数后验完全坍缩)


@dataclass
class MALIBOConfig:
    """MALIBO 综合全局配置"""

    resffn: ResFFNConfig = field(default_factory=ResFFNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    booster: BoosterConfig = field(default_factory=BoosterConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
