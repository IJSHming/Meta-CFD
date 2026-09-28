"""MALIBO 任务自适应 (Task Adaptation) 与拉普拉斯近似采样模块.

严格遵循 Pan et al. (2024, Section 4 & Appendix G) 官方算法流程：
1. Eq. (6): 固定元学习特征提取器 g_omega，通过 L-BFGS 快速优化目标任务嵌入 z_MAP；
2. Eq. (7): 基于严格 Hessian 矩阵构建后验协方差矩阵 Sigma_N 的拉普拉斯高斯近似 q(z) = N(z_MAP, Sigma_N)；
3. Eq. (9): 执行汤普森采样 (Thompson Sampling) 抽取候选任务向量 z_hat ~ q(z)，提供不确定度驱动的高效探索。
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from malibo.config import AdaptationConfig
from malibo.models import MALIBOMetaModel


class LaplaceTaskAdapter:
    """基于 Laplace 逼近与 Thompson 采样的目标任务快速自适应器."""

    def __init__(
        self,
        meta_model: MALIBOMetaModel,
        config: Optional[AdaptationConfig] = None,
        gamma: float = 1.0 / 3.0,
        device: Optional[torch.device] = None,
    ):
        self.meta_model = meta_model
        self.config = config if config is not None else AdaptationConfig()
        self.gamma = float(gamma)

        if device is None:
            self.device = next(meta_model.parameters()).device
        else:
            self.device = device

        self.feature_dim = meta_model.feature_dim

    def estimate_z_map(
        self,
        Phi: torch.Tensor,
        m: torch.Tensor,
        y: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        """利用 L-BFGS 求解式 (6) 的 MAP 任务嵌入向量 z_MAP."""
        d = self.feature_dim
        z = torch.zeros(d, dtype=torch.float32, device=self.device, requires_grad=True)

        # 预计算标签 k 与效用权重 w
        k = (y <= tau).float()
        w = torch.clamp(tau - y, min=0.0)

        # 严格依据 Appendix G 配置 L-BFGS 优化器
        optimizer = torch.optim.LBFGS(
            [z],
            lr=self.config.lr,
            max_iter=self.config.max_iter,
            tolerance_grad=self.config.tolerance_grad,
            tolerance_change=self.config.tolerance_change,
            history_size=self.config.history_size,
            line_search_fn=self.config.line_search_fn,
        )

        def closure():
            optimizer.zero_grad()
            # a_n = m(Phi_n) + z^T Phi_n
            a = m.squeeze(-1) + (Phi @ z)
            # 似然项: w * softplus(-a) + softplus(a)
            nll = (w * F.softplus(-a) + F.softplus(a)).sum()
            # 标准高斯先验正则: 0.5 * ||z||_2^2
            prior = 0.5 * torch.sum(z**2)
            loss = prior + nll
            loss.backward()
            return loss

        optimizer.step(closure)
        return z.detach()

    def compute_laplace_cholesky(
        self,
        z_map: torch.Tensor,
        Phi: torch.Tensor,
        m: torch.Tensor,
        y: torch.Tensor,
        tau: float,
    ) -> torch.Tensor:
        """依据式 (7) 构建负 Hessian 精度矩阵 Sigma_N^-1 并求解其 Cholesky 分解矩阵 L.

        Sigma_N^-1 = I_d + sum_{n=1}^N (k_n * (tau - y_n) + 1) * k_hat_n * (1 - k_hat_n) * Phi_n * Phi_n^T
        """
        d = self.feature_dim
        a_map = m.squeeze(-1) + (Phi @ z_map)
        k_hat = torch.sigmoid(a_map)  # (N,)

        k = (y <= tau).float()
        # 权重标量: (k_n * (tau - y_n) + 1) * k_hat * (1 - k_hat)
        weights_laplace = (k * (tau - y) + 1.0) * k_hat * (1.0 - k_hat)  # (N,)

        # 高性能矩阵乘法构建精度矩阵: I_d + Phi^T * diag(weights) * Phi
        weighted_Phi = Phi * weights_laplace.unsqueeze(-1)
        precision_matrix = torch.eye(d, dtype=torch.float32, device=self.device) + (
            weighted_Phi.T @ Phi
        )

        # 转换为 float64 并强制对称化，消除 float32 舍入带来的微小非对称误差
        prec_64 = precision_matrix.to(torch.float64)
        prec_64 = 0.5 * (prec_64 + prec_64.T)

        # 动态自适应对角抖动 (Adaptive Jitter)
        diag_scale = torch.clamp(torch.diag(prec_64), min=1.0)
        jitter = torch.diag(diag_scale * 1e-6)

        try:
            L_64 = torch.linalg.cholesky(prec_64 + jitter)
        except torch.linalg.LinAlgError:
            # 极端异常值保护: 特征值分解截断极小或负特征值
            eigvals, eigvecs = torch.linalg.eigh(prec_64)
            eigvals = torch.clamp(eigvals, min=1e-5)
            prec_psd = eigvecs @ torch.diag(eigvals) @ eigvecs.T
            L_64 = torch.linalg.cholesky(prec_psd)

        return L_64.to(torch.float32)

    def sample_thompson_z(
        self,
        z_map: torch.Tensor,
        L_precision: torch.Tensor,
        exploration_floor: float = 0.15,
    ) -> torch.Tensor:
        """从后验分布 q(z) = N(z_MAP, Sigma_N) 中高效抽取单次 Thompson 采样向量 z_hat.

        利用 L @ L^T = Sigma_N^-1，可知 Sigma_N = L^-T @ L^-1.
        引入 exploration_floor 防止高步数下精度矩阵特征值爆炸导致后验方差完全坍缩 (Variance Collapse)
        进而退化为纯贪婪剥削。
        """
        d = self.feature_dim
        eps = torch.randn(d, 1, dtype=torch.float32, device=self.device)
        delta = torch.linalg.solve_triangular(
            L_precision.T, eps, upper=True
        ).squeeze(-1)
        if exploration_floor > 0:
            delta = delta + exploration_floor * torch.randn(d, dtype=torch.float32, device=self.device)
        z_sample = z_map + delta
        return z_sample

    def adapt_and_sample(
        self,
        x_observed: torch.Tensor,
        y_observed: torch.Tensor,
        gamma: Optional[float] = None,
        exploration_floor: float = 0.15,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """完整任务自适应流程：从目标任务观测集中推断后验并抽取一个 Thompson 任务嵌入.

        返回:
            (z_sample, z_map) 形状均为 (d,)
        """
        x_observed = x_observed.to(self.device)
        y_observed = y_observed.to(self.device)

        # 1. 提取固定特征 Phi 与均值预测 m
        with torch.no_grad():
            Phi, m = self.meta_model.extract_features(x_observed)

        # 2. 计算分位数阈值 tau (支持动态退火 gamma)
        effective_gamma = float(gamma) if gamma is not None else self.gamma
        tau = float(torch.quantile(y_observed, effective_gamma).item())

        # 3. 求解 MAP 任务嵌入
        z_map = self.estimate_z_map(Phi, m, y_observed, tau)

        # 4. 计算精度矩阵 Cholesky 分解
        L_prec = self.compute_laplace_cholesky(z_map, Phi, m, y_observed, tau)

        # 5. 抽取 Thompson 样本
        z_sample = self.sample_thompson_z(z_map, L_prec, exploration_floor=exploration_floor)

        return z_sample, z_map
