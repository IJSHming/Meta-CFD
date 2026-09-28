"""MALIBO 决策树残差梯度提升模型 (Residual Gradient Boosting).

严格遵循 Pan et al. (2024, Section 4 Eq. 10 & Appendix G) 官方机制：
- 将 Meta-Model 的 Thompson 采样概率作为基分类器 (Base Estimator / Init Estimator)；
- 针对目标任务的在线观测数据，通过梯度提升弱分类器集成 sum_{i=1}^M r_i(x) 学习残差修正；
- 采用 70% 训练 / 30% 验证集自动确定早停最佳树数量，彻底杜绝小样本过拟合。
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
    """供 scikit-learn GradientBoostingClassifier 使用的强基学习器.

    在初始步输出 Meta-Model 与抽样任务嵌入 z_hat 预测的后验概率:
    p(k=1|x) = sigmoid(m(phi(x)) + z_hat^T phi(x)).
    """

    def __init__(
        self,
        meta_model: MALIBOMetaModel,
        z_sample: torch.Tensor,
        device: Optional[torch.device] = None,
    ):
        self.meta_model = meta_model
        self.z_sample = z_sample
        self.device = device if device is not None else next(meta_model.parameters()).device
        self.classes_ = np.array([0, 1], dtype=np.int32)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, sample_weight: Optional[np.ndarray] = None):
        """无需拟合，权重来自预训练 Meta-Model 与任务自适应嵌入."""
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """输出形状为 (N, 2) 的 [p(k=0|x), p(k=1|x)]."""
        tX = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            p1 = self.meta_model.predict_proba(tX, z=self.z_sample).detach().cpu().numpy().flatten()

        # 数值裁剪防 logit 溢出 (使用 1e-12 扩增动态范围)
        p1 = np.clip(p1, 1e-12, 1.0 - 1e-12)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


class MALIBOResidualBooster:
    """残差梯度提升树封装管理器 (Eq. 10)."""

    def __init__(
        self,
        meta_model: MALIBOMetaModel,
        config: Optional[BoosterConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.meta_model = meta_model
        self.config = config if config is not None else BoosterConfig()
        self.device = device

        self.booster: Optional[GradientBoostingClassifier] = None
        self.is_fitted = False
        self.current_z_sample: Optional[torch.Tensor] = None

    def fit_with_early_stopping(
        self,
        X: np.ndarray,
        y_minimize: np.ndarray,
        tau: float,
        z_sample: torch.Tensor,
    ) -> bool:
        """训练残差梯度提升树.

        参数:
            X: (N, D) 已观测设计点
            y_minimize: (N,) 已观测目标值 (越小越好)
            tau: 分位数阈值
            z_sample: (d,) 当前 Thompson 抽取的任务嵌入

        返回:
            bool: 是否成功拟合残差模型
        """
        N = len(X)
        self.current_z_sample = z_sample
        k = (y_minimize <= tau).astype(np.int32)

        # 检查是否包含两类标签 (二分类树必须至少有一个正例和一个负例)
        unique_classes = np.unique(k)
        if len(unique_classes) < 2:
            self.is_fitted = False
            self.booster = None
            return False

        # 计算样本效用加权 (Eq. 2 LFBO 加权)
        raw_weights = np.maximum(tau - y_minimize, 0.0)
        # 归一化样本权重避免树梯度/Hessian 在极端差距下产生数值爆炸
        std_y = float(np.std(y_minimize))
        scale = max(std_y, 1.0)
        norm_weights = np.clip(raw_weights / scale * 5.0, 0.0, 20.0)
        sample_weights = np.where(k == 1, norm_weights + 1.0, 1.0)

        init_estimator = MALIBOInitialEstimator(
            meta_model=self.meta_model,
            z_sample=z_sample,
            device=self.device,
        )

        # 依据 Appendix G: 70% 训练 / 30% 验证早停确定树数量
        if N >= 10:
            try:
                X_tr, X_val, k_tr, k_val, w_tr, w_val = train_test_split(
                    X, k, sample_weights, test_size=self.config.val_ratio, random_state=42, stratify=k
                )
            except ValueError:
                # 若类别极度不均衡无法分层，使用普通切分
                X_tr, X_val, k_tr, k_val, w_tr, w_val = train_test_split(
                    X, k, sample_weights, test_size=self.config.val_ratio, random_state=42
                )

            max_depth = getattr(self.config, "max_depth", 3)
            pilot_clf = GradientBoostingClassifier(
                init=init_estimator,
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                max_depth=max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=42,
            )
            pilot_clf.fit(X_tr, k_tr, sample_weight=w_tr)

            val_losses = []
            for y_pred in pilot_clf.staged_predict_proba(X_val):
                val_losses.append(log_loss(k_val, y_pred, sample_weight=w_val))

            best_trees = int(np.argmin(val_losses) + 1)
        else:
            best_trees = min(10, self.config.n_estimators)

        max_depth = getattr(self.config, "max_depth", 3)
        # 在全量观测数据上训练最终残差模型
        final_clf = GradientBoostingClassifier(
            init=init_estimator,
            n_estimators=max(best_trees, 1),
            learning_rate=self.config.learning_rate,
            max_depth=max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=42,
        )
        final_clf.fit(X, k, sample_weight=sample_weights)

        self.booster = final_clf
        self.is_fitted = True
        return True

    def predict_utility(
        self,
        X_candidates: np.ndarray,
        z_sample: torch.Tensor,
        use_raw_margin: bool = True,
    ) -> np.ndarray:
        """计算候选集的最终采集效用 C_GB(x; z_hat) (Eq. 10 / Eq. 21).

        返回连续 logit 决策边界评分 (decision_function)，彻底杜绝传统 predict_proba
        在 16~20 以上 logit 时发生 1.0000 概率饱和截断，并在树模型叶子平原内
        融合元神经网络的连续梯度，保障微米级精细寻优。
        """
        device = self.device if self.device is not None else next(self.meta_model.parameters()).device
        if self.is_fitted and self.booster is not None:
            if use_raw_margin:
                return self.booster.decision_function(X_candidates)
            else:
                return self.booster.predict_proba(X_candidates)[:, 1]
        else:
            tX = torch.as_tensor(X_candidates, dtype=torch.float32, device=device)
            with torch.no_grad():
                Phi, m = self.meta_model.extract_features(tX)
                z_term = (Phi @ z_sample.to(device)).squeeze(-1)
                return (m.squeeze(-1) + z_term).detach().cpu().numpy()
