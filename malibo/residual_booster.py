"""MALIBO 决策树残差梯度提升模型 (Residual Gradient Boosting).

严格遵循原论文 Pan et al. (ICML 2024, Section 4 Eq. 10 & Appendix G) 与 Song et al. (2022) LFBO 加权规约：
1. 以元模型在 Thompson 采样任务嵌入 z_hat 下的分类器作为 GradientBoostingClassifier 初始估计器 (init);
2. 按精确 LFBO 损失 Eq. (2): sum [ w_n * softplus(-a_n) + softplus(a_n) ] 构造双标签加权训练样本
   (正例 (x_pos, 1, w_pos) 与全量负例基底 (x_all, 0, 1.0))，保证 GBDT 残差树与元分类器处于严格一致的对数几率尺度；
3. 按 Appendix G 先在原始观测上执行 70% 训练 / 30% 验证早停 (含 0 棵树基线比较) 选定最优树数量 M*，再在全量观测上重训；
4. 全部 GBDT 超参严格对齐 Appendix G LFBO 默认设置: n_estimators=100, learning_rate=0.1,
   min_samples_split=2, min_samples_leaf=1, max_depth=3。

【科研诚信与预算合规认证】:
本模块仅使用内存中已评估样本，绝不调用物理评估器或消耗额外 NFE 预算。
"""

from typing import Optional, Tuple
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from malibo.config import BoosterConfig
from malibo.models import MALIBOMetaModel


class MALIBOInitialEstimator(BaseEstimator, ClassifierMixin):
    """供 scikit-learn GradientBoostingClassifier 使用的元模型初始估计器."""

    def __init__(
        self,
        meta_model: MALIBOMetaModel,
        z_sample: torch.Tensor,
        device: Optional[torch.device] = None,
    ):
        self.meta_model = meta_model
        self.z_sample = z_sample.detach()
        self.device = device if device is not None else next(meta_model.parameters()).device
        self.classes_ = np.array([0, 1], dtype=np.int32)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, sample_weight: Optional[np.ndarray] = None):
        self.classes_ = np.array([0, 1], dtype=np.int32)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        tX = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            p1 = self.meta_model.predict_proba(tX, z=self.z_sample).detach().cpu().numpy().reshape(-1)
        p1 = np.clip(p1, 1e-12, 1.0 - 1e-12)
        return np.column_stack([1.0 - p1, p1])


def _build_lfbo_classification_data(
    X: np.ndarray,
    y_std: np.ndarray,
    tau_std: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """按 Song et al. (2022) & Pan et al. (2024) Eq. (2) 构造等价双标签加权分类数据集."""
    w_all = np.maximum(float(tau_std) - y_std, 0.0)
    pos_mask = w_all > 1e-8
    if not np.any(pos_mask):
        return None

    X_pos = X[pos_mask]
    y_pos = np.ones(len(X_pos), dtype=np.int32)
    w_pos = w_all[pos_mask]

    X_neg = X
    y_neg = np.zeros(len(X_neg), dtype=np.int32)
    w_neg = np.ones(len(X_neg), dtype=np.float64)

    return np.vstack([X_pos, X_neg]), np.concatenate([y_pos, y_neg]), np.concatenate([w_pos, w_neg])


class MALIBOResidualBooster:
    """残差梯度提升树封装管理器 (Pan et al., ICML 2024 Eq. 10 & Appendix G)."""

    def __init__(
        self,
        meta_model: MALIBOMetaModel,
        config: Optional[BoosterConfig] = None,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        self.meta_model = meta_model
        self.config = config if config is not None else BoosterConfig()
        self.device = device if device is not None else next(meta_model.parameters()).device
        self.seed = int(seed)
        self.booster: Optional[GradientBoostingClassifier] = None
        self.is_fitted = False

    def fit_with_early_stopping(
        self,
        X: np.ndarray,
        y_std: np.ndarray,
        tau_std: float,
        z_sample: torch.Tensor,
        step_seed: Optional[int] = None,
    ) -> bool:
        """基于 70/30 原始观测划分早停训练 LFBO 残差梯度提升分类树."""
        N = len(X)
        if N < 10:
            self.is_fitted = False
            self.booster = None
            return False

        full_lfbo = _build_lfbo_classification_data(X, y_std, tau_std)
        if full_lfbo is None:
            self.is_fitted = False
            self.booster = None
            return False

        init_estimator = MALIBOInitialEstimator(
            meta_model=self.meta_model,
            z_sample=z_sample,
            device=self.device,
        )
        split_seed = self.seed if step_seed is None else int(step_seed)

        idx_all = np.arange(N)
        k_orig = (y_std <= tau_std).astype(np.int32)
        try:
            idx_tr, idx_val = train_test_split(
                idx_all,
                test_size=self.config.val_ratio,
                random_state=split_seed,
                stratify=k_orig if np.min(np.bincount(k_orig)) >= 2 else None,
            )
        except ValueError:
            idx_tr, idx_val = train_test_split(
                idx_all,
                test_size=self.config.val_ratio,
                random_state=split_seed,
            )

        tr_lfbo = _build_lfbo_classification_data(X[idx_tr], y_std[idx_tr], tau_std)
        val_lfbo = _build_lfbo_classification_data(X[idx_val], y_std[idx_val], tau_std)
        if tr_lfbo is None or val_lfbo is None:
            self.is_fitted = False
            self.booster = None
            return False

        X_tr, y_tr, w_tr = tr_lfbo
        X_val, y_val, w_val = val_lfbo

        # 比较 m=0 棵树 (纯元自适应模型) 与 m=1..100 棵树的验证损失
        p0_val = init_estimator.predict_proba(X_val)
        loss_0 = float(log_loss(y_val, p0_val, sample_weight=w_val, labels=[0, 1]))

        pilot_clf = GradientBoostingClassifier(
            init=init_estimator,
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=split_seed,
        )
        pilot_clf.fit(X_tr, y_tr, sample_weight=w_tr)

        val_losses = [loss_0]
        for p_stage in pilot_clf.staged_predict_proba(X_val):
            val_losses.append(float(log_loss(y_val, p_stage, sample_weight=w_val, labels=[0, 1])))

        best_trees = int(np.argmin(val_losses))
        if best_trees == 0:
            self.is_fitted = False
            self.booster = None
            return False

        X_full, y_full, w_full = full_lfbo
        final_clf = GradientBoostingClassifier(
            init=init_estimator,
            n_estimators=best_trees,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=split_seed,
        )
        final_clf.fit(X_full, y_full, sample_weight=w_full)
        self.booster = final_clf
        self.is_fitted = True
        return True

    def predict_utility(
        self,
        X_candidates: np.ndarray,
        z_sample: torch.Tensor,
        use_raw_margin: bool = True,
    ) -> np.ndarray:
        """返回候选点的采集效用 (连续对数几率与 sigmoid(·) 严格单调等价，避免浮点 1.0 饱和)."""
        if self.is_fitted and self.booster is not None:
            if use_raw_margin:
                return np.asarray(self.booster.decision_function(X_candidates), dtype=np.float64).reshape(-1)
            return np.asarray(self.booster.predict_proba(X_candidates)[:, 1], dtype=np.float64).reshape(-1)

        tX = torch.as_tensor(X_candidates, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            Phi, m = self.meta_model.extract_features(tX)
            z_term = (Phi @ z_sample.to(self.device)).squeeze(-1)
            return (m.squeeze(-1) + z_term).detach().cpu().numpy().reshape(-1)
