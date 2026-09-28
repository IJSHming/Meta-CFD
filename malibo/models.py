"""MALIBO 神经网络架构实现 (ResFFN + MeanHead + Task Embeddings).

严格遵循 Pan et al. (2024, Figure 2 & Appendix G) 的网络结构：
- ResFFN: 4 个残差块，每层 64 单元，ELU 激活，映射到 d=50 维特征空间 Phi = phi(x)；
- MeanHead: 任务无关均值预测层 m(Phi) -> R；
- Task Embedding: 各任务独立低维嵌入向量 z_t in R^d，结合先验分布 p(Z) = N(0, I)；
- 联合输出分类 Logit: a = m(phi(x)) + z_t^T phi(x)。
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from malibo.config import ResFFNConfig


class ResBlock(nn.Module):
    """标准残差全连接块 (Residual Feedforward Block)."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        return self.act(out + residual)


class ResFFN(nn.Module):
    """残差前馈特征映射网络 phi(x): R^D -> R^d.

    将原始 D 维设计参数 (如 18 维 Kulfan 参数) 映射至共享潜在特征空间 Phi in R^50。
    """

    def __init__(self, config: Optional[ResFFNConfig] = None):
        super().__init__()
        if config is None:
            config = ResFFNConfig()

        self.in_dim = config.in_dim
        self.hidden_dim = config.hidden_dim
        self.feature_dim = config.feature_dim

        # 初始投影层
        self.input_proj = nn.Linear(self.in_dim, self.hidden_dim)
        self.act = nn.ELU()

        # 4 个连续残差块
        self.res_blocks = nn.ModuleList(
            [ResBlock(self.hidden_dim) for _ in range(config.num_hidden_layers)]
        )

        # 映射至 d 维特征空间
        self.output_proj = nn.Linear(self.hidden_dim, self.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播: 输入 (B, in_dim) -> 输出 (B, feature_dim)."""
        h = self.act(self.input_proj(x))
        for block in self.res_blocks:
            h = block(h)
        phi = self.output_proj(h)
        return phi


class MeanHead(nn.Module):
    """任务无关均值预测层 m(Phi): R^d -> R.

    学习所有相关任务共享的高效用区域先验，在零观察步为贝叶斯优化提供首选初始点 x_0。
    """

    def __init__(self, feature_dim: int = 50, hidden_dim: int = 50):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.act = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """输出形状: (B, 1)."""
        h = self.act(self.fc1(phi))
        return self.fc2(h)


class MALIBOMetaModel(nn.Module):
    """MALIBO 完整元学习分类器网络.

    封装 task-agnostic 骨干 g_omega = {phi, m} 与 task-specific 嵌入集合 {z_t}_{t=1}^T。
    """

    def __init__(
        self,
        num_tasks: int,
        config: Optional[ResFFNConfig] = None,
    ):
        super().__init__()
        if config is None:
            config = ResFFNConfig()

        self.num_tasks = num_tasks
        self.feature_dim = config.feature_dim

        # 1. 任务无关核心 (Task-agnostic components g_omega)
        self.phi = ResFFN(config)
        self.mean_head = MeanHead(feature_dim=self.feature_dim, hidden_dim=50)

        # 2. 任务特定嵌入矩阵 Z in R^(T x d)，服从标准正态先验 N(0, I)
        # 用 scale 0.1 初始化，后续通过 KS+Cov 正则项约束至标准高斯分布
        self.task_embeddings = nn.Parameter(
            torch.randn(num_tasks, self.feature_dim) * 0.1
        )

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取特征表示 Phi 与均值预测 m(Phi).

        返回: (Phi, m_phi) 形状分别为 (B, d) 和 (B, 1).
        """
        phi = self.phi(x)
        m_phi = self.mean_head(phi)
        return phi, m_phi

    def forward(
        self,
        x: torch.Tensor,
        task_idx: torch.Tensor,
    ) -> torch.Tensor:
        """元训练前向计算 Logit: a = m(Phi) + z_t^T Phi.

        参数:
            x: (B, in_dim) 批量输入
            task_idx: (B,) 各样本对应的任务索引 (0 <= task_idx < T)

        返回:
            logits: (B, 1)
        """
        phi, m_phi = self.extract_features(x)
        # 索引对应任务的嵌入向量 z_t: (B, d)
        z_t = self.task_embeddings[task_idx]

        # 任务线性预测层: h_zt(Phi) = sum_j (z_t,j * Phi_j)
        h_zt = (z_t * phi).sum(dim=-1, keepdim=True)
        logits = m_phi + h_zt
        return logits

    def predict_proba(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """针对指定任务嵌入 z 计算分类后验概率 p(k=1|x) = sigmoid(m(Phi) + z^T Phi).

        若 z 为 None，则退化为任务无关均值概率预测 sigma(m(Phi)).
        """
        phi, m_phi = self.extract_features(x)
        if z is None:
            logits = m_phi
        else:
            if z.ndim == 1:
                z = z.unsqueeze(0)  # (1, d)
            logits = m_phi + (z * phi).sum(dim=-1, keepdim=True)
        return torch.sigmoid(logits)

    def predict_mean_utility(self, x: torch.Tensor) -> torch.Tensor:
        """计算任务无关均值效用 g_omega(x) = sigma(m(phi(x))), 供初始候选 x_0 选取使用."""
        return self.predict_proba(x, z=None)
