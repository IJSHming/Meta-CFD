"""MALIBO 损失函数与潜在空间高斯正则化器实现.

严格遵循 Pan et al. (2024) 论文公式：
1. LFBOLoss: Eq. (2) & Eq. (6) 基于 softplus 的全数值稳定分类与效用加权损失；
2. LatentSpaceRegularizer: Eq. (4) & Appendix C 中的 Kolmogorov-Smirnov (KS)
   边际分布匹配与经验协方差正则项，自动校准 lambda_KS 和 lambda_Cov 系数。
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LFBOLoss(nn.Module):
    """无似然贝叶斯优化 (LFBO) 效用加权损失函数 (Eq. 2 & Eq. 6).

    数学推导证明:
      - [w * ln(sigma(a)) + ln(1 - sigma(a))]
      = w * softplus(-a) + softplus(a)
    这种形式彻底规避了 log(0)、除以零与极大梯度问题，数学上严格连续、凸且处处可导。
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """计算批次样本平均 LFBO 损失.

        参数:
            logits: (B, 1) 或 (B,) 网络输出 Logit 值 a
            weights: (B, 1) 或 (B,) 效用权重 w = max(tau - y, 0)
        """
        logits = logits.view(-1)
        weights = weights.view(-1)

        # 全局数值稳定计算
        loss_pos = weights * F.softplus(-logits)
        loss_neg = F.softplus(logits)
        total_loss = loss_pos + loss_neg

        return total_loss.mean()


class LatentSpaceRegularizer(nn.Module):
    """潜在任务空间多维高斯先验正则化器 (Eq. 4 & Appendix C, Eq. 28~29).

    将所有任务嵌入集合 {z_t}_{t=1}^T 约束至标准多维正态先验 p(Z) = N(0, I_d):
    R({z_t}; p(Z)) = lambda_KS * L_KS + lambda_Cov * L_Cov
    """

    def __init__(
        self,
        feature_dim: int = 50,
        tau_smooth: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.tau_smooth = tau_smooth
        self.register_buffer("identity", torch.eye(feature_dim))

        # 缓存动态标定权重
        self.lambda_ks: Optional[float] = None
        self.lambda_cov: Optional[float] = None

    def _compute_ks_loss(self, Z: torch.Tensor) -> torch.Tensor:
        """计算边际累积分布函数 (CDF) 差异的平方和 (Eq. 27 & 28 第一项)."""
        T, d = Z.shape
        # 使用平滑 Sigmoid 核函数计算可微经验 CDF: F(z_t, j)
        diff = Z.unsqueeze(1) - Z.unsqueeze(0)  # (T, T, d)
        F_emp = torch.sigmoid(diff / self.tau_smooth).mean(dim=1)  # (T, d)

        # 标准正态理论分布 CDF: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
        Phi_std = 0.5 * (1.0 + torch.erf(Z * (1.0 / 1.4142135623730951)))

        # 汇总各维度的均方误差
        ks_loss = torch.sum((F_emp - Phi_std) ** 2) / T
        return ks_loss

    def _compute_cov_loss(self, Z: torch.Tensor) -> torch.Tensor:
        """计算经验协方差矩阵与单位阵 I_d 的 Frobenius 范数误差 (Eq. 28 第二项)."""
        T, d = Z.shape
        Z_centered = Z - Z.mean(dim=0, keepdim=True)
        cov = (Z_centered.T @ Z_centered) / max(T - 1, 1)
        if self.identity.shape == cov.shape and self.identity.device == cov.device:
            identity = self.identity
        else:
            identity = torch.eye(d, dtype=cov.dtype, device=cov.device)
        cov_loss = torch.sum((cov - identity) ** 2)
        return cov_loss

    def calibrate_coefficients(
        self,
        num_tasks: int,
        device: torch.device,
        num_samples: int = 500,
    ) -> Tuple[float, float]:
        """依据论文 Eq. (29) 通过从标准正态分布中采样估算归一化系数 lambda_KS 与 lambda_Cov."""
        with torch.no_grad():
            Z_prior = torch.randn(num_tasks, self.feature_dim, device=device)
            base_ks = self._compute_ks_loss(Z_prior).item()
            base_cov = self._compute_cov_loss(Z_prior).item()

            # lambda^-1 = 2 * L_prior
            self.lambda_ks = 1.0 / (2.0 * max(base_ks, 1e-6))
            self.lambda_cov = 1.0 / (2.0 * max(base_cov, 1e-6))

        return self.lambda_ks, self.lambda_cov

    def forward(self, Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算综合正则化损失.

        返回: (R_total, ks_loss, cov_loss)
        """
        if self.lambda_ks is None or self.lambda_cov is None:
            self.calibrate_coefficients(Z.shape[0], Z.device)

        ks_loss = self._compute_ks_loss(Z)
        cov_loss = self._compute_cov_loss(Z)

        R_total = self.lambda_ks * ks_loss + self.lambda_cov * cov_loss
        return R_total, ks_loss, cov_loss
