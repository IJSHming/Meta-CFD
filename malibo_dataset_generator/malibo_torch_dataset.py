"""面向 MALIBO 神经网络训练的 PyTorch Dataset 与数据加载器。

严格依照 MALIBO (Pan et al., 2024) 论文公式设计：
- Eq. (2) & Eq. (3): 基于 gamma 分位数 (默认 1/3) 自动划分高低效用子集并计算样本权重 w = max(tau - y, 0)。
- Eq. (4): 支持跨任务多任务元批次 (Meta-Batch) 采样与任务索引映射，直接适配 ResFFN 特征网络与任务嵌入层 z_t 联合反向传播。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class MALIBOTaskData:
    """单一任务在 PyTorch 体系下的结构化张量封装"""

    task_id: int
    task_name: str
    airfoil_name: str
    reynolds: float
    alpha: float
    x_unit: torch.Tensor             # (N, 18) 归一化输入参数 [0, 1]^18
    theta: torch.Tensor              # (N, 18) 物理 Kulfan 参数
    y_minimize: torch.Tensor         # (N,) 最小化目标值 y = -L/D_penalized
    y_penalized: torch.Tensor        # (N,) 真实带惩罚升阻比
    tau: float                       # 分位数阈值 tau = Phi^-1(gamma)
    labels: torch.Tensor             # (N,) 二分类标签 k = 1(y <= tau)
    weights: torch.Tensor            # (N,) LFBO 效用权重 w = max(tau - y, 0)
    cl: torch.Tensor                 # (N,) 升力系数
    cd: torch.Tensor                 # (N,) 阻力系数
    cm: torch.Tensor                 # (N,) 俯仰力矩
    is_feasible: torch.Tensor        # (N,) 几何可行性掩码 (bool)
    num_samples: int                 # 样本条数 N (通常为 500)


class MALIBOMetaDataset(Dataset):
    """MALIBO 多任务元学习数据集。

    提供两种标准访问模式：
    1. 展平模式 (Flat Sample Mode): 将所有任务的样本拼接，__len__() 返回 N_total，每次返回 (x, task_idx, label, weight, y)；
    2. 任务结构模式 (Task-Structured Mode): 按任务索引访问，方便逐任务计算梯度与匹配高斯先验正则项 R({z_t})。
    """

    def __init__(
        self,
        tasks_dir: Optional[Union[str, Path]] = None,
        bundled_pt_path: Optional[Union[str, Path]] = None,
        gamma: float = 1.0 / 3.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.gamma = float(gamma)
        self.device = device
        self.tasks: List[MALIBOTaskData] = []

        # 展平索引缓存
        self._flat_x: Optional[torch.Tensor] = None
        self._flat_task_idx: Optional[torch.Tensor] = None
        self._flat_labels: Optional[torch.Tensor] = None
        self._flat_weights: Optional[torch.Tensor] = None
        self._flat_y: Optional[torch.Tensor] = None

        if bundled_pt_path and Path(bundled_pt_path).exists():
            self.load_bundled(bundled_pt_path)
        elif tasks_dir and Path(tasks_dir).exists():
            self.load_from_tasks_dir(tasks_dir)

    def load_from_tasks_dir(self, tasks_dir: Union[str, Path]) -> None:
        """从分散的 task_*.npz 文件集中加载所有元任务数据"""
        tasks_dir = Path(tasks_dir)
        npz_files = sorted(tasks_dir.glob("task_*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"未在目录 {tasks_dir} 下找到任何 task_*.npz 数据文件!")

        self.tasks = []
        for fpath in npz_files:
            data = np.load(fpath)

            x_unit = torch.tensor(data["X_unit"], dtype=torch.float32)
            theta = torch.tensor(data["Theta"], dtype=torch.float32)
            y_penalized = torch.tensor(data["Y_penalized"], dtype=torch.float32)
            y_min = torch.tensor(data["Y_minimize"], dtype=torch.float32)

            # 计算该任务特异性的 gamma 分位数阈值 tau
            # 注意: y_min 越小越好 (y = -L/D)，gamma 分位数代表最优秀的 top-gamma 样本
            tau_val = float(np.quantile(y_min.numpy(), self.gamma))

            # 二分类标签: k = 1 (y <= tau，即属于 top 1/3 的优质设计)
            labels = (y_min <= tau_val).float()

            # 效用加权项: w = max(tau - y, 0)
            weights = torch.clamp(tau_val - y_min, min=0.0)

            task_obj = MALIBOTaskData(
                task_id=int(data["task_id"]),
                task_name=fpath.stem,
                airfoil_name=str(data["airfoil_name"]),
                reynolds=float(data["reynolds"]),
                alpha=float(data["alpha"]),
                x_unit=x_unit,
                theta=theta,
                y_minimize=y_min,
                y_penalized=y_penalized,
                tau=tau_val,
                labels=labels,
                weights=weights,
                cl=torch.tensor(data["CL"], dtype=torch.float32),
                cd=torch.tensor(data["CD"], dtype=torch.float32),
                cm=torch.tensor(data["CM"], dtype=torch.float32),
                is_feasible=torch.tensor(data["is_feasible"], dtype=torch.bool),
                num_samples=len(x_unit),
            )
            self.tasks.append(task_obj)

        self._rebuild_flat_indices()

    def _rebuild_flat_indices(self) -> None:
        """重构展平张量，以便支持随机 Shuffle DataLoader"""
        x_list = []
        task_idx_list = []
        label_list = []
        weight_list = []
        y_list = []

        for t_idx, t in enumerate(self.tasks):
            x_list.append(t.x_unit)
            task_idx_list.append(torch.full((t.num_samples,), t_idx, dtype=torch.long))
            label_list.append(t.labels)
            weight_list.append(t.weights)
            y_list.append(t.y_minimize)

        self._flat_x = torch.cat(x_list, dim=0)
        self._flat_task_idx = torch.cat(task_idx_list, dim=0)
        self._flat_labels = torch.cat(label_list, dim=0)
        self._flat_weights = torch.cat(weight_list, dim=0)
        self._flat_y = torch.cat(y_list, dim=0)

    @property
    def num_tasks(self) -> int:
        """任务总数 T"""
        return len(self.tasks)

    @property
    def total_samples(self) -> int:
        """全部样本总数 (例如 175 x 500 = 87,500)"""
        return len(self._flat_x) if self._flat_x is not None else 0

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回单个样本点: (x_unit, task_idx, label, weight, y_minimize)"""
        return (
            self._flat_x[idx],
            self._flat_task_idx[idx],
            self._flat_labels[idx],
            self._flat_weights[idx],
            self._flat_y[idx],
        )

    def get_task(self, task_idx: int) -> MALIBOTaskData:
        """获取指定任务的完整数据包"""
        return self.tasks[task_idx]

    def sample_meta_batch(
        self,
        batch_size_per_task: int = 32,
        task_indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """抽取多任务均衡批次 (Meta-Batch)，供 MALIBO 论文联合更新 omega 与 {z_t} 使用。"""
        if task_indices is None:
            task_indices = list(range(self.num_tasks))

        bx, bt, bl, bw, by = [], [], [], [], []
        for t_idx in task_indices:
            t = self.tasks[t_idx]
            sub_idx = torch.randint(0, t.num_samples, (batch_size_per_task,))
            bx.append(t.x_unit[sub_idx])
            bt.append(torch.full((batch_size_per_task,), t_idx, dtype=torch.long))
            bl.append(t.labels[sub_idx])
            bw.append(t.weights[sub_idx])
            by.append(t.y_minimize[sub_idx])

        return {
            "x": torch.cat(bx, dim=0).to(self.device),
            "task_idx": torch.cat(bt, dim=0).to(self.device),
            "labels": torch.cat(bl, dim=0).to(self.device),
            "weights": torch.cat(bw, dim=0).to(self.device),
            "y": torch.cat(by, dim=0).to(self.device),
        }

    def save_bundled(self, save_path: Union[str, Path]) -> None:
        """将整个元数据集打包保存为单个 .pt 文件，实现秒级加载"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "gamma": self.gamma,
            "tasks": self.tasks,
            "flat_x": self._flat_x,
            "flat_task_idx": self._flat_task_idx,
            "flat_labels": self._flat_labels,
            "flat_weights": self._flat_weights,
            "flat_y": self._flat_y,
        }
        torch.save(payload, save_path)
        print(f">>> [MALIBOMetaDataset] 已成功打包保存全集至: {save_path} ({self.total_samples:,} 条数据)")

    def load_bundled(self, load_path: Union[str, Path]) -> None:
        """从预打包 .pt 文件中极速还原数据集"""
        payload = torch.load(load_path, map_location="cpu", weights_only=False)
        self.gamma = float(payload["gamma"])
        self.tasks = payload["tasks"]
        self._flat_x = payload["flat_x"]
        self._flat_task_idx = payload["flat_task_idx"]
        self._flat_labels = payload["flat_labels"]
        self._flat_weights = payload["flat_weights"]
        self._flat_y = payload["flat_y"]
        print(f">>> [MALIBOMetaDataset] 已极速加载预打包数据集: {load_path} ({self.total_samples:,} 条数据)")
