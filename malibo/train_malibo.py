"""MALIBO 神经网络元训练命令行入口脚本 (train_malibo.py).

用法示例:
    1. 使用生成的 87,500 条气动数据执行全量元训练:
       python -m malibo.train_malibo --dataset malibo_dataset_generator/data/malibo_train_dataset.pt --epochs 2048

    2. 快速测试训练流程 (跑 5 轮快速验证):
       python -m malibo.train_malibo --dataset malibo_dataset_generator/data/malibo_train_dataset.pt --epochs 5 --log-interval 1
"""

import argparse
import sys
import time
from pathlib import Path

# 确保控制台输出 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from malibo.config import ResFFNConfig, TrainingConfig
from malibo.models import MALIBOMetaModel
from malibo.trainer import MALIBOTrainer
from malibo_dataset_generator.malibo_torch_dataset import MALIBOMetaDataset


def main():
    parser = argparse.ArgumentParser(
        description="MALIBO 元学习神经网络训练脚本 (Pan et al., 2024)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="malibo_dataset_generator/data/malibo_train_dataset.pt",
        help="MALIBO 训练集 .pt 文件路径 (默认: malibo_dataset_generator/data/malibo_train_dataset.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2048,
        help="最大训练轮数 (默认: 2048)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam 初始学习率 (默认: 1e-3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="批大小 (默认: 256)",
    )
    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=0.1,
        help="任务空间高斯正则化系数 (默认: 0.1, 论文 Eq. 3)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=80,
        help="早停轮数 (默认: 80)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_PACKAGE_DIR / "malibo_checkpoint.pt"),
        help="模型权重导出路径 (默认: malibo/malibo_checkpoint.pt)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="控制台日志打印间隔 (轮数, 默认: 50)",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用 torch.compile(mode='reduce-overhead') CUDA Graphs 极速固化训练 (默认: True)",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"未找到数据集文件: {dataset_path}！请先确认数据已生成。"
        )

    print(f">>> 正在加载 MALIBO 预打包训练集: {dataset_path}...")
    dataset = MALIBOMetaDataset(bundled_pt_path=dataset_path)
    print(
        f"    - 成功加载 {dataset.num_tasks} 个元任务, 累计 {dataset.total_samples:,} 条气动评估样本!"
    )

    # 模型架构初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_cfg = ResFFNConfig(in_dim=18, feature_dim=50)
    model = MALIBOMetaModel(num_tasks=dataset.num_tasks, config=res_cfg)

    # 训练配置
    train_cfg = TrainingConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.patience,
        reg_lambda=args.reg_lambda,
        use_compile=args.compile,
    )

    trainer = MALIBOTrainer(model=model, config=train_cfg, device=device)

    # 执行训练
    output_path = Path(args.output)
    history = trainer.train(
        train_dataset=dataset,
        checkpoint_path=output_path,
        log_interval=args.log_interval,
    )

    print(f"\n>>> 训练完成! 最佳模型权重已保存至: {output_path}")


if __name__ == "__main__":
    main()
