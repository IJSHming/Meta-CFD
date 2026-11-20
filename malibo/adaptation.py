"""MALIBO 任务自适应 (Task Adaptation) 与拉普拉斯近似采样模块.

严格遵循原论文 Pan et al. (ICML 2024, Section 4, Eq. 6-9 & Appendix G)：
1. Eq. (6): 固定元学习特征提取器 g_omega，通过 L-BFGS 优化目标任务嵌入 z_MAP；
2. Eq. (7): 计算精确负 Hessian 精度矩阵 Sigma_N^-1 = I_d + sum (w_n + 1) k_hat_n (1 - k_hat_n) Phi_n Phi_n^T；
3. Eq. (9): 执行纯净拉普拉斯 Thompson 采样 z_hat ~ N(z_MAP, Sigma_N) (固定 gamma=1/3, 无人工方差地板)。
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from malibo.config import AdaptationConfig
from malibo.models import MALIBOMetaModel


class LaplaceTaskAdapter:
    """基于 Laplace 逼近与 Thompson 采样的目标任务快速自适应器 (Strictly Paper-Exact Eq. 6-9)."""

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
        self.meta_model.eval()
        for p in self.meta_model.parameters():
            p.requires_grad = False

    def estimate_z_map(
        self,
        Phi: torch.Tensor,
        m: torch.Tensor,
        y_std: torch.Tensor,
        tau_std: float,
    ) -> torch.Tensor:
        """利用 L-BFGS 求解 Eq. (6) 的 MAP 任务嵌入向量 z_MAP."""
        d = self.feature_dim
        z = torch.zeros(d, dtype=torch.float32, device=self.device, requires_grad=True)
        w = torch.clamp(float(tau_std) - y_std, min=0.0)

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
            a = m.squeeze(-1) + (Phi @ z)
            nll = (w * F.softplus(-a) + F.softplus(a)).sum()
            prior = 0.5 * torch.sum(z ** 2)
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
        y_std: torch.Tensor,
        tau_std: float,
    ) -> torch.Tensor:
        """依据 Eq. (7) 构建精确负 Hessian 精度矩阵 Sigma_N^-1 并求其 Cholesky 分解 L."""
        d = self.feature_dim
        a_map = m.squeeze(-1) + (Phi @ z_map)
        k_hat = torch.sigmoid(a_map)  # (N,)
        w = torch.clamp(float(tau_std) - y_std, min=0.0)

        weights_laplace = (w + 1.0) * k_hat * (1.0 - k_hat)  # (N,)
        weighted_Phi = Phi * weights_laplace.unsqueeze(-1)
        precision_matrix = torch.eye(d, dtype=torch.float32, device=self.device) + (
            weighted_Phi.T @ Phi
        )

        prec_64 = precision_matrix.to(torch.float64)
        prec_64 = 0.5 * (prec_64 + prec_64.T)
        diag_scale = torch.clamp(torch.diag(prec_64), min=1.0)
        jitter = torch.diag(diag_scale * 1e-6)

        try:
            L_64 = torch.linalg.cholesky(prec_64 + jitter)
        except torch.linalg.LinAlgError:
            eigvals, eigvecs = torch.linalg.eigh(prec_64)
            eigvals = torch.clamp(eigvals, min=1e-5)
            prec_psd = eigvecs @ torch.diag(eigvals) @ eigvecs.T
            L_64 = torch.linalg.cholesky(prec_psd)

        return L_64.to(torch.float32)

    def sample_thompson_z(
        self,
        z_map: torch.Tensor,
        L_precision: torch.Tensor,
        rng: Optional[np.random.RandomState] = None,
    ) -> torch.Tensor:
        """从后验 q(z) = N(z_MAP, Sigma_N) 中执行纯净 Thompson 采样 (Eq. 9, 无人工方差地板)."""
        d = self.feature_dim
        eps_np = rng.randn(d, 1) if rng is not None else np.random.randn(d, 1)
        eps = torch.as_tensor(eps_np, dtype=torch.float32, device=self.device)
        delta = torch.linalg.solve_triangular(L_precision.T, eps, upper=True).squeeze(-1)
        return z_map + delta

    def adapt_and_sample(
        self,
        x_observed: torch.Tensor,
        y_observed_std: torch.Tensor,
        tau_std: Optional[float] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """完整任务自适应流程：从标准化目标观测中求解后验并抽取 Thompson 样本 (z_sample, z_map)."""
        x_observed = x_observed.to(self.device)
        y_observed_std = y_observed_std.to(self.device)

        with torch.no_grad():
            Phi, m = self.meta_model.extract_features(x_observed)

        tau_val = float(tau_std) if tau_std is not None else float(
            torch.quantile(y_observed_std, self.gamma).item()
        )
        z_map = self.estimate_z_map(Phi, m, y_observed_std, tau_val)
        L_prec = self.compute_laplace_cholesky(z_map, Phi, m, y_observed_std, tau_val)
        z_sample = self.sample_thompson_z(z_map, L_prec, rng=rng)
        return z_sample, z_map
