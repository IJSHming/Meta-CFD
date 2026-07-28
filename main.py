import argparse

import run


def main():
    parser = argparse.ArgumentParser(
        description="基于强化学习 (SAC) 的翼型元优化实验控制台工具"
    )

    # 建立子命令 (train 或 test)
    subparsers = parser.add_subparsers(
        dest="command", help="可选操作: train (训练并保存) 或 test (加载并预测评估)"
    )
    subparsers.required = True

    # 1. 训练子命令: python main.py train
    train_parser = subparsers.add_parser("train", help="训练 SAC 模型并保存到本地")
    train_parser.add_argument(
        "--model_path",
        type=str,
        default="sac_metaspsa",
        help="模型保存的文件名称/路径 (默认: sac_metaspsa)",
    )
    train_parser.add_argument(
        "--timesteps",
        type=int,
        default=5000,
        help="SAC 元训练的总 Timesteps 步数 (默认: 5000)",
    )

    # 2. 测试子命令: python main.py test
    test_parser = subparsers.add_parser(
        "test", help="加载已保存的模型进行预测评估与图形绘制"
    )
    test_parser.add_argument(
        "--model_path",
        type=str,
        default="sac_metaspsa",
        help="需要加载的模型文件名称/路径 (默认: sac_metaspsa)",
    )

    args = parser.parse_args()

    # 路由分发
    if args.command == "train":
        run.train_and_save(model_path=args.model_path, timesteps=args.timesteps)
    elif args.command == "test":
        run.load_and_test(model_path=args.model_path)


if __name__ == "__main__":
    main()
