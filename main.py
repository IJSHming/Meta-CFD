"""命令行入口文件"""

import argparse

import run


def main():
    parser = argparse.ArgumentParser(
        description="基于 AeroSandbox / NeuralFoil / SAC 的翼型 CMA-ES 元优化系统"
    )

    subparsers = parser.add_subparsers(
        dest="command", help="可选命令: train (训练) 或 test (评估)"
    )
    subparsers.required = True

    # 1. 训练命令
    train_parser = subparsers.add_parser(
        "train", help="训练 SAC 元优化模型并保存至本地"
    )
    train_parser.add_argument(
        "--model_path",
        type=str,
        default="sac_metacmaes",
        help="模型保存路径 (默认: sac_metacmaes)",
    )
    train_parser.add_argument(
        "-n",
        "--rounds_per_combo",
        type=int,
        default=1,
        help="每种组合的训练轮数 (默认: 1)",
    )

    # 2. 测试命令
    test_parser = subparsers.add_parser(
        "test", help="对指定翼型及工况进行 10 次独立测试评估"
    )
    test_parser.add_argument(
        "--model_path",
        type=str,
        default="sac_metacmaes",
        help="模型加载路径 (默认: sac_metacmaes)",
    )
    test_parser.add_argument(
        "--airfoil",
        type=str,
        required=True,
        help="起始翼型名称 (例如: naca2412, naca0012 等)",
    )
    test_parser.add_argument(
        "--reynolds",
        type=float,
        required=True,
        help="测试雷诺数 (例如: 1e6 或 2.5e6)",
    )
    test_parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="测试攻角 Alpha (单位: deg, 例如: 5.0)",
    )

    args = parser.parse_args()

    if args.command == "train":
        run.train_and_save(
            model_path=args.model_path,
            rounds_per_combo=args.rounds_per_combo,
        )
    elif args.command == "test":
        run.load_and_test(
            model_path=args.model_path,
            airfoil_name=args.airfoil,
            reynolds=args.reynolds,
            alpha=args.alpha,
        )


if __name__ == "__main__":
    main()
