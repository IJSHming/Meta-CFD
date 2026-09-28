"""MALIBO 神经网络元训练引擎 (Meta-Trainer).

严格遵循 Pan et al. (2024, Section 4 Eq. 3 & Appendix G) 训练机制：
- 联合最小化 LFBO 效用分类损失与任务空间高斯先验正则项 R({z_t}; p(Z))；
- 采用 Adam 优化器，学习率 lr=1e-3，每轮指数衰减 factor=0.999；
- 批大小 B=256，支持早停 (Early Stopping) 与模型权重检查点自动保存。
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from malibo.config import TrainingConfig
from malibo.losses import LatentSpaceRegularizer, LFBOLoss
from malibo.models import MALIBOMetaModel


class MALIBOTrainer:
    """MALIBO 元学习分类器训练管理器."""

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
            "loss": [],
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
        """执行完整 2,048 轮 (带早停) 元训练循环.

        参数:
            train_dataset: PyTorch Dataset (如 MALIBOMetaDataset)
            checkpoint_path: 最优模型保存路径
            log_interval: 控制台日志打印间隔 (轮数)
        """
        # 判断是否可开启显存常驻 (In-VRAM Zero-Copy) 极速批处理
        use_in_vram = (
            self.device.type == "cuda"
            and hasattr(train_dataset, "_flat_x")
            and train_dataset._flat_x is not None
        )

        if use_in_vram:
            print("    - 数据加载模式: 显存常驻极速切片 (In-VRAM Zero-Copy, 零 CPU 开销)")
            x_all = train_dataset._flat_x.to(self.device, non_blocking=True)
            task_idx_all = train_dataset._flat_task_idx.to(self.device, non_blocking=True)
            weights_all = train_dataset._flat_weights.to(self.device, non_blocking=True)
            total_samples = len(x_all)
            batch_size = self.config.batch_size
            num_batches_per_epoch = total_samples // batch_size
            n_used = num_batches_per_epoch * batch_size
        else:
            print("    - 数据加载模式: 标准 PyTorch DataLoader (CPU 主存流式批次)")
            loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=(self.device.type == "cuda"),
            )

        # 优化器配置: CUDA 环境优先开启 Fused Adam，合并内核消除启动延迟
        use_fused = (
            getattr(self.config, "fused_adam", True)
            and self.device.type == "cuda"
        )
        optimizer = Adam(self.model.parameters(), lr=self.config.lr, fused=use_fused)
        scheduler = ExponentialLR(optimizer, gamma=self.config.lr_decay_factor)

        # 单批训练原子步骤
        def _train_step(bx, b_task_idx, bw):
            optimizer.zero_grad(set_to_none=True)
            logits = self.model(bx, b_task_idx)
            loss_lfbo = self.lfbo_criterion(logits, bw)
            loss_reg, ks, cov = self.reg_criterion(self.model.task_embeddings)
            total_loss = loss_lfbo + self.config.reg_lambda * loss_reg
            total_loss.backward()
            optimizer.step()
            return total_loss, loss_lfbo, loss_reg, ks, cov

        # 计算图编译优化 (CUDA Graphs): 消除 Python 调度与内核发射开销
        use_compile = (
            getattr(self.config, "use_compile", True)
            and self.device.type == "cuda"
            and hasattr(torch, "compile")
        )
        if use_compile:
            print("    - 算子优化: 启用 torch.compile(mode='reduce-overhead') CUDA Graphs 极速固化")
            train_step_fn = torch.compile(_train_step, mode="reduce-overhead")
        else:
            train_step_fn = _train_step

        # 标定 KS 与 Cov 正则权重系数
        self.reg_criterion.calibrate_coefficients(
            num_tasks=self.model.num_tasks, device=self.device
        )

        best_loss = float("inf")
        patience_counter = 0
        t0 = time.time()

        print("=" * 80)
        print(f">>> [MALIBOTrainer] 开始元训练 (Meta-Training)...")
        print(f"    - 设备: {self.device}")
        print(f"    - 训练样本总量: {len(train_dataset):,} 条")
        print(f"    - 任务数量: {self.model.num_tasks} 个")
        print(f"    - 批大小: {self.config.batch_size}, 最大轮数: {self.config.epochs}")
        print(f"    - 正则化系数: lambda = {self.config.reg_lambda}")
        print("=" * 80)

        for epoch in range(1, self.config.epochs + 1):
            epoch_t0 = time.time()
            self.model.train()
            epoch_lfbo = 0.0
            epoch_reg = 0.0
            epoch_ks = 0.0
            epoch_cov = 0.0
            epoch_total = 0.0
            num_batches = 0

            if use_in_vram:
                # 显存内生成整批随机置换，单次切片重排
                perm = torch.randperm(total_samples, device=self.device)[:n_used]
                xs = x_all[perm]
                ts = task_idx_all[perm]
                ws = weights_all[perm]

                for i in range(0, n_used, batch_size):
                    bx = xs[i : i + batch_size]
                    b_task_idx = ts[i : i + batch_size]
                    bw = ws[i : i + batch_size]

                    total_loss, loss_lfbo, loss_reg, ks, cov = train_step_fn(bx, b_task_idx, bw)

                    epoch_lfbo += loss_lfbo.item()
                    epoch_reg += loss_reg.item()
                    epoch_ks += ks.item()
                    epoch_cov += cov.item()
                    epoch_total += total_loss.item()
                    num_batches += 1
            else:
                for batch in loader:
                    bx, b_task_idx, _, bw, _ = batch
                    bx = bx.to(self.device, non_blocking=True)
                    b_task_idx = b_task_idx.to(self.device, non_blocking=True)
                    bw = bw.to(self.device, non_blocking=True)

                    total_loss, loss_lfbo, loss_reg, ks, cov = train_step_fn(bx, b_task_idx, bw)

                    epoch_lfbo += loss_lfbo.item()
                    epoch_reg += loss_reg.item()
                    epoch_ks += ks.item()
                    epoch_cov += cov.item()
                    epoch_total += total_loss.item()
                    num_batches += 1

            scheduler.step()

            avg_loss = epoch_total / max(num_batches, 1)
            avg_lfbo = epoch_lfbo / max(num_batches, 1)
            avg_reg = epoch_reg / max(num_batches, 1)
            avg_ks = epoch_ks / max(num_batches, 1)
            avg_cov = epoch_cov / max(num_batches, 1)
            current_lr = scheduler.get_last_lr()[0]

            self.history["epoch"].append(epoch)
            self.history["loss"].append(avg_loss)
            self.history["lfbo_loss"].append(avg_lfbo)
            self.history["reg_loss"].append(avg_reg)
            self.history["ks_loss"].append(avg_ks)
            self.history["cov_loss"].append(avg_cov)
            self.history["lr"].append(current_lr)

            # 打印日志
            if epoch % log_interval == 0 or epoch == 1 or epoch == self.config.epochs:
                elapsed = time.time() - t0
                epoch_ms = (time.time() - epoch_t0) * 1000.0
                print(
                    f"Epoch [{epoch:4d}/{self.config.epochs:4d}] "
                    f"| Loss: {avg_loss:.4f} (LFBO: {avg_lfbo:.4f}, Reg: {avg_reg:.4f}) "
                    f"| KS: {avg_ks:.4f}, Cov: {avg_cov:.4f} "
                    f"| LR: {current_lr:.2e} | Speed: {epoch_ms:.1f}ms/ep | Total: {elapsed:.1f}s"
                )

            # 评估与早停检测
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
                if checkpoint_path is not None:
                    self.save_checkpoint(checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print(
                        f"\n>>> [Early Stopping] 连续 {self.config.early_stopping_patience} 轮损失未下降，在第 {epoch} 轮触发早停!"
                    )
                    break

        print(f"\n>>> [MALIBOTrainer] 训练完成! 最佳 Loss: {best_loss:.4f}")
        return self.history

    def save_checkpoint(self, path: Path) -> None:
        """保存模型全量权重及元信息."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "num_tasks": self.model.num_tasks,
                "feature_dim": self.model.feature_dim,
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        """加载已保存的检查点权重."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "history" in checkpoint:
            self.history = checkpoint["history"]
