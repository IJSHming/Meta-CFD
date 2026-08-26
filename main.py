"""命令行入口文件"""

import argparse
import sys

# 确保在 Windows 控制台环境下 UTF-8 字符正确输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import run


def main():
    parser = argparse.ArgumentParser(
        description="基于 AeroSandbox / NeuralFoil / TuRBO 的翼型信任域贝叶斯优化系统"
    )

    # 兼容直接输入参数或使用 test 子命令
    subparsers = parser.add_subparsers(dest="command", help="可选子命令: test")

    test_parser = subparsers.add_parser(
        "test", help="对指定翼型及工况进行 10 次独立 TuRBO 测试评估"
    )

    for p in [parser, test_parser]:
        p.add_argument(
            "--airfoil",
            type=str,
            default="naca2412",
            help="起始翼型名称 (默认: naca2412, 支持: naca0012, naca4412 等)",
        )
        p.add_argument(
            "--reynolds",
            type=float,
            default=2.5e6,
            help="测试雷诺数 Re (默认: 2.5e6)",
        )
        p.add_argument(
            "--alpha",
            type=float,
            default=5.0,
            help="测试攻角 Alpha (单位: deg, 默认: 5.0)",
        )
        p.add_argument(
            "-n",
            "--runs",
            type=int,
            default=10,
            help="独立优化测试轮数 (默认: 10)",
        )

    args = parser.parse_args()

    run.test_turbo(
        airfoil_name=args.airfoil,
        reynolds=args.reynolds,
        alpha=args.alpha,
        n_runs=args.runs,
    )


if __name__ == "__main__":
    main()
