"""MALIBO: Meta-learning for Likelihood-free Bayesian Optimization (PyTorch 实现).

严格遵循 Pan et al. (2024, arXiv:2307.03565v3) 论文数学机理构建：
- 共享残差特征网络 (ResFFN) + 任务无关均值层 m(phi)
- 高斯先验正则化潜在任务空间 (KS 检验 + 协方差匹配)
- 基于拉普拉斯近似与汤普森采样 (Thompson Sampling) 的高效任务自适应
- 决策树残差梯度提升模型 (Gradient Boosting Residual Model)
"""

__version__ = "1.0.0"

from malibo.config import (
    AdaptationConfig,
    BoosterConfig,
    MALIBOConfig,
    OptimizationConfig,
    ResFFNConfig,
    TrainingConfig,
)
from malibo.models import MALIBOMetaModel, ResFFN
from malibo.optimizer import MALIBOOptimizer
from malibo.trainer import MALIBOTrainer

__all__ = [
    "ResFFNConfig",
    "TrainingConfig",
    "AdaptationConfig",
    "BoosterConfig",
    "OptimizationConfig",
    "MALIBOConfig",
    "ResFFN",
    "MALIBOMetaModel",
    "MALIBOTrainer",
    "MALIBOOptimizer",
]
