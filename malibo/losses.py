"""MALIBO 损失函数与潜在空间高斯正则化器实现.

严格遵循原论文 Pan et al. (ICML 2024) 数学公式：
1. LFBOLoss: Eq. (2) & Eq. (6) 基于 softplus 的数值稳定分类与效用加权损失:
   L_LFBO = E[ w * softplus(-a) + softplus(a) ], w = max(tau - y, 0);
2. LatentSpaceRegularizer: Eq. (4) & Appendix C (Eq. 27-29) 基于排序顺序统计量的精确非参数
   Kolmogorov-Smirnov (KS) 边际 CDF 匹配与经验协方差矩阵 Frobenius 范数匹配正则项。
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LFBOLoss(nn.Module):
    """无似然贝叶斯优化 (LFBO) 效用加权损失函数 (Eq. 2 & Eq. 6)."""

    def forward(
        self,
        logits: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        logits = logits.view(-1)
        weights = weights.view(-1)
        return (weights * F.softplus(-logits) + F.softplus(logits)).mean()


class LatentSpaceRegularizer(nn.Module):
    """潜在任务空间多维高斯先验正则化器 (Eq. 4 & Appendix C, Eq. 27-29).

    将所有任务嵌入集合 {z_t}_{t=1}^T 约束至标准多维正态先验 p(Z) = N(0, I_d):
      R({z_t}; p(Z)) = lambda_KS * L_KS + lambda_Cov * L_Cov
    """

    def __init__(
        self,
        feature_dim: int = 50,
        tau_smooth: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.tau_smooth = float(tau_smooth)
        self.register_buffer("identity", torch.eye(self.feature_dim))
        self.lambda_ks: Optional[float] = None
        self.lambda_cov: Optional[float] = None

    def _compute_ks_loss(self, Z: torch.Tensor) -> torch.Tensor:
        """精确非参数边际经验 CDF 与标准正态 CDF 均方误差 (Eq. 27 & Eq. 28 第一项).

        对每维任务嵌入排序后，第 i 个顺序统计量的经验 CDF 为 (i - 0.5) / T，
        梯度通过可微标准正态 CDF Phi(z_(i)) 精确回传至 Z，杜绝平滑核导致的原点塌缩。
        """
        T, _ = Z.shape
        z_sorted, _ = torch.sort(Z, dim=0)  # (T, d), differentiable wrt Z
        ranks = (torch.arange(1, T + 1, dtype=Z.dtype, device=Z.device) - 0.5) / float(T)
        F_emp = ranks.unsqueeze(1)  # (T, 1) -> broadcasts to (T, d)
        Phi_std = 0.5 * (1.0 + torch.erf(z_sorted * 0.7071067811865475))
        return torch.sum((F_emp - Phi_std) ** 2) / max(T, 1)

    def _compute_cov_loss(self, Z: torch.Tensor) -> torch.Tensor:
        """计算经验协方差矩阵与单位阵 I_d 的 Frobenius 范数平方误差 (Eq. 28 第二项)."""
        T, d = Z.shape
        Z_centered = Z - Z.mean(dim=0, keepdim=True)
        cov = (Z_centered.T @ Z_centered) / max(T - 1, 1)
        if self.identity.shape == cov.shape and self.identity.device == cov.device:
            identity = self.identity
        else:
            identity = torch.eye(d, dtype=cov.dtype, device=cov.device)
        return torch.sum((cov - identity) ** 2)

    def calibrate_coefficients(
        self,
        num_tasks: int,
        device: torch.device,
        num_samples: int = 200,
    ) -> Tuple[float, float]:
        """依据论文 Eq. (29) 多次采样 N(0, I_d) 估算归一化系数 lambda_KS 与 lambda_Cov."""
        ks_vals = []
        cov_vals = []
        with torch.no_grad():
            for _ in range(max(1, num_samples)):
                Z_prior = torch.randn(num_tasks, self.feature_dim, device=device)
                ks_vals.append(self._compute_ks_loss(Z_prior).item())
                cov_vals.append(self._compute_cov_loss(Z_prior).item())
            base_ks = float(sum(ks_vals) / len(ks_vals))
            base_cov = float(sum(cov_vals) / len(cov_vals))
            self.lambda_ks = 1.0 / (2.0 * max(base_ks, 1e-6))
            self.lambda_cov = 1.0 / (2.0 * max(base_cov, 1e-6))
        return self.lambda_ks, self.lambda_cov

    def forward(self, Z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.lambda_ks is None or self.lambda_cov is None:
            self.calibrate_coefficients(Z.shape[0], Z.device)
        ks_loss = self._compute_ks_loss(Z)
        cov_loss = self._compute_cov_loss(Z)
        R_total = self.lambda_ks * ks_loss + self.lambda_cov * cov_loss
        return R_total, ks_loss, cov_loss
