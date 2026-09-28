"""MALIBO Dataset Generator Package.

面向 MALIBO (Meta-learning for Likelihood-free Bayesian Optimization) 神经网络训练集的
高通量自动化数据生成与管理系统。
"""

__version__ = "1.0.0"

from malibo_dataset_generator.cma_es_optimizer import (
    OptimizationTrajectory,
    run_cma_es,
)

__all__ = ["OptimizationTrajectory", "run_cma_es"]
