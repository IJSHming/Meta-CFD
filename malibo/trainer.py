"""MALIBO 神经网络元训练引擎 (Meta-Trainer).

严格遵循原论文 Pan et al. (ICML 2024, Section 4 Eq. 2-4 & Appendix C/G)：
- 联合最小化精确 Eq. (2) LFBO 效用分类损失与 Eq. (4) 排序 KS + 协方差高斯先验正则项 R({z_t}; N(0, I))；
- 采用 Adam 优化器 (lr=1e-3, 无 weight_decay)，每轮指数衰减 factor=0.999；
- 批大小 B=256，80/20 训练-验证集切分并基于验证损失执行早停 (Early Stopping)。
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, random_split

from malibo.config import TrainingConfig
from malibo.losses import LatentSpaceRegularizer, LFBOLoss
from malibo.models import MALIBOMetaModel


class MALIBOTrainer:
    """MALIBO 元学习分类器训练管理器 (Strictly Paper-Exact Appendix G)."""

    def __init__(
        self,
        model: MALIBOMetaModel,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config if config is not None else TrainingConfig()

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() and self.config.device == "cuda" else "cpu"
            )
        else:
            self.device = device

        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")

        self.model.to(self.device)
        self.lfbo_criterion = LFBOLoss()
        self.reg_criterion = LatentSpaceRegularizer(
            feature_dim=self.model.feature_dim
        ).to(self.device)

        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "lfbo_loss": [],
            "reg_loss": [],
            "ks_loss": [],
            "cov_loss": [],
            "lr": [],
        }

    def train(
        self,
        train_dataset,
        checkpoint_path: Optional[Path] = None,
        log_interval: int = 50,
    ) -> Dict[str, List[float]]:
        """执行 80/20 验证早停元训练循环 (最多 2,048 轮)."""
        total_samples = len(train_dataset)
        val_len = int(total_samples * 0.20)
        train_len = total_samples - val_len

        gen = torch.Generator().manual_seed(self.config.seed)
        train_sub, val_sub = random_split(train_dataset, [train_len, val_len], generator=gen)

        use_in_vram = (
            self.device.type == "cuda"
            and hasattr(train_dataset, "_flat_x")
            and train_dataset._flat_x is not None
        )

        batch_size = self.config.batch_size
        if use_in_vram:
            tr_idx = torch.tensor(train_sub.indices, dtype=torch.long)
            va_idx = torch.tensor(val_sub.indices, dtype=torch.long)
            x_tr_all = train_dataset._flat_x[tr_idx].to(self.device, non_blocking=True)
            t_tr_all = train_dataset._flat_task_idx[tr_idx].to(self.device, non_blocking=True)
            w_tr_all = train_dataset._flat_weights[tr_idx].to(self.device, non_blocking=True)

            x_va_all = train_dataset._flat_x[va_idx].to(self.device, non_blocking=True)
            t_va_all = train_dataset._flat_task_idx[va_idx].to(self.device, non_blocking=True)
            w_va_all = train_dataset._flat_weights[va_idx].to(self.device, non_blocking=True)
        else:
            train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, drop_last=False)
            val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, drop_last=False)

        # Appendix G: Adam(lr=1e-3, 无 weight_decay) + ExponentialLR(0.999)
        optimizer = Adam(self.model.parameters(), lr=self.config.lr)
        scheduler = ExponentialLR(optimizer, gamma=self.config.lr_decay_factor)

        # 标定 KS 与 Cov 正则权重系数 (Eq. 29)
        l_ks, l_cov = self.reg_criterion.calibrate_coefficients(
            num_tasks=self.model.num_tasks, device=self.device, num_samples=200
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        t0 = time.time()
        in_dim = int(train_dataset._flat_x.shape[-1]) if hasattr(train_dataset, "_flat_x") else 18

        print("=" * 84)
        print(f">>> [MALIBOTrainer] 开始严格原论文版元训练 (Device={self.device}, Tasks={self.model.num_tasks})")
        print(f"    - 样本划分: Train={train_len:,}, Val={val_len:,} | Batch={batch_size}, MaxEpochs={self.config.epochs}")
        print(f"    - 正则标定: lambda={self.config.reg_lambda}, lambda_KS={l_ks:.4f}, lambda_Cov={l_cov:.4f}")
        print("=" * 84)

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            tr_total = 0.0
            tr_lfbo = 0.0
            n_tr_batches = 0

            if use_in_vram:
                perm = torch.randperm(len(x_tr_all), device=self.device)
                for i in range(0, len(x_tr_all), batch_size):
                    idx_b = perm[i : i + batch_size]
                    bx, bt, bw = x_tr_all[idx_b], t_tr_all[idx_b], w_tr_all[idx_b]
                    optimizer.zero_grad(set_to_none=True)
                    logits = self.model(bx, bt)
                    loss_lfbo = self.lfbo_criterion(logits, bw)
                    loss_reg, _, _ = self.reg_criterion(self.model.task_embeddings)
                    total_loss = loss_lfbo + self.config.reg_lambda * loss_reg
                    total_loss.backward()
                    optimizer.step()
                    tr_total += total_loss.item()
                    tr_lfbo += loss_lfbo.item()
                    n_tr_batches += 1
            else:
                for batch in train_loader:
                    bx, bt, _, bw = batch[0].to(self.device), batch[1].to(self.device), batch[2], batch[3].to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = self.model(bx, bt)
                    loss_lfbo = self.lfbo_criterion(logits, bw)
                    loss_reg, _, _ = self.reg_criterion(self.model.task_embeddings)
                    total_loss = loss_lfbo + self.config.reg_lambda * loss_reg
                    total_loss.backward()
                    optimizer.step()
                    tr_total += total_loss.item()
                    tr_lfbo += loss_lfbo.item()
                    n_tr_batches += 1

            scheduler.step()

            # 验证集评估
            self.model.eval()
            va_total = 0.0
            n_va_batches = 0
            with torch.no_grad():
                loss_reg_eval, ks_eval, cov_eval = self.reg_criterion(self.model.task_embeddings)
                reg_val = float(loss_reg_eval.item())
                ks_val = float(ks_eval.item())
                cov_val = float(cov_eval.item())

                if use_in_vram:
                    for i in range(0, len(x_va_all), batch_size):
                        bx, bt, bw = x_va_all[i : i + batch_size], t_va_all[i : i + batch_size], w_va_all[i : i + batch_size]
                        logits = self.model(bx, bt)
                        va_total += (self.lfbo_criterion(logits, bw) + self.config.reg_lambda * loss_reg_eval).item()
                        n_va_batches += 1
                else:
                    for batch in val_loader:
                        bx, bt, bw = batch[0].to(self.device), batch[1].to(self.device), batch[3].to(self.device)
                        logits = self.model(bx, bt)
                        va_total += (self.lfbo_criterion(logits, bw) + self.config.reg_lambda * loss_reg_eval).item()
                        n_va_batches += 1

            avg_tr = tr_total / max(n_tr_batches, 1)
            avg_va = va_total / max(n_va_batches, 1)
            avg_lfbo = tr_lfbo / max(n_tr_batches, 1)
            current_lr = scheduler.get_last_lr()[0]

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(avg_tr)
            self.history["val_loss"].append(avg_va)
            self.history["lfbo_loss"].append(avg_lfbo)
            self.history["reg_loss"].append(reg_val)
            self.history["ks_loss"].append(ks_val)
            self.history["cov_loss"].append(cov_val)
            self.history["lr"].append(current_lr)

            if avg_va < best_val_loss - 1e-4:
                best_val_loss = avg_va
                patience_counter = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if epoch % log_interval == 0 or epoch == 1 or patience_counter >= self.config.early_stopping_patience:
                elapsed = time.time() - t0
                print(
                    f"Epoch [{epoch:4d}/{self.config.epochs:4d}] "
                    f"| Train: {avg_tr:.4f} | Val: {avg_va:.4f} (Best: {best_val_loss:.4f}) "
                    f"| R={reg_val:.4f} (KS={ks_val:.4f}, Cov={cov_val:.4f}) "
                    f"| LR: {current_lr:.2e} | {elapsed:.1f}s",
                    flush=True,
                )

            if patience_counter >= self.config.early_stopping_patience:
                print(f">>> [Early Stopping] 连续 {self.config.early_stopping_patience} 轮验证损失未下降，在第 {epoch} 轮早停!")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        with torch.no_grad():
            z_emb = self.model.task_embeddings.detach().cpu()
            z_std = float(z_emb.std().item())
            z_mean_abs = float(z_emb.mean(dim=0).abs().mean().item())
        print(f">>> [MALIBOTrainer] 训练完成! Best Val Loss={best_val_loss:.4f}, z_t: std={z_std:.4f}, |mean|={z_mean_abs:.4f}")

        if checkpoint_path is not None:
            self.save_checkpoint(checkpoint_path, in_dim=in_dim, best_val_loss=best_val_loss)

        return self.history

    def save_checkpoint(self, path: Path, in_dim: int = 18, best_val_loss: float = 0.0) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "num_tasks": self.model.num_tasks,
                "feature_dim": self.model.feature_dim,
                "hidden_dim": self.model.phi.hidden_dim,
                "num_hidden_layers": len(self.model.phi.res_blocks),
                "in_dim": in_dim,
                "best_val_loss": best_val_loss,
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "history" in checkpoint:
            self.history = checkpoint["history"]
