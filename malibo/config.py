"""MALIBO 架构、元训练、任务自适应与优化超参数配置。

严格对齐原论文 Pan et al. (ICML 2024, Section 4, Algorithm 1 & Appendix G) 全部原始超参数，
彻底移除信赖域 (Trust Region)、稀疏坐标掩码、gamma 退火、探索方差地板与几何预筛器等非原论文扩展。
"""

from dataclasses import dataclass, field


@dataclass
class ResFFNConfig:
    """特征提取残差前馈网络 (ResFFN) 结构超参 (Appendix G)"""

    in_dim: int = 18                 # 输入参数维度 (Kulfan 18 维连续参数)
    hidden_dim: int = 64             # 隐层神经元数量 (4 层，每层 64 单元)
    num_hidden_layers: int = 4       # 残差块数量
    feature_dim: int = 50            # 潜在特征空间维度 d (论文固定为 50)
    activation: str = "elu"          # 激活函数 (ELU，对齐 Tiao et al. 2021)


@dataclass
class TrainingConfig:
    """元训练 (Meta-Training) 超参数配置 (Appendix G)"""

    lr: float = 1e-3                 # Adam 优化器初始学习率 (无 weight_decay)
    lr_decay_factor: float = 0.999   # 学习率指数衰减因子 (每 Epoch 乘以 0.999)
    batch_size: int = 256            # 批大小 B = 256
    epochs: int = 2048               # 最大训练轮数 2,048 (带验证早停)
    early_stopping_patience: int = 80
    reg_lambda: float = 0.1          # 任务空间高斯先验正则化权重 lambda = 0.1 (Eq. 3)
    gamma: float = 1.0 / 3.0         # LFBO 固定分位数比例 gamma = 1/3 (Eq. 2)
    seed: int = 42
    device: str = "cuda"


@dataclass
class AdaptationConfig:
    """目标任务快速自适应 (Task Adaptation) L-BFGS 超参数 (Appendix G)"""

    lr: float = 1.0
    max_iter: int = 20
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    history_size: int = 100
    line_search_fn: str = "strong_wolfe"


@dataclass
class BoosterConfig:
    """残差梯度提升分类器 (Residual Gradient Boosting) 配置 (Eq. 10 & Appendix G, LFBO defaults)"""

    n_estimators: int = 100
    learning_rate: float = 0.1
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_depth: int = 3
    val_ratio: float = 0.30          # Appendix G: 70% train / 30% validation 早停


@dataclass
class OptimizationConfig:
    """贝叶斯元优化运行配置 (Algorithm 1 & Appendix G: 全局 5,120 随机采样，无信赖域)"""

    n_candidates: int = 5120         # Appendix G: 全局 [0, 1]^D 上 5,120 个均匀随机候选点
    budget: int = 300                # 测试优化步数上限
    seed: int = 42                   # 优化器随机种子
    n_init: int = 15                 # 初始设计点数 (与对比基准共享统一初始化协议)
    gamma: float = 1.0 / 3.0         # Appendix G: 固定分位数比例 gamma = 1/3 (无退火)


@dataclass
class MALIBOConfig:
    """MALIBO 综合全局配置"""

    resffn: ResFFNConfig = field(default_factory=ResFFNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)
    booster: BoosterConfig = field(default_factory=BoosterConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
