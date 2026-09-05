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
        description="基于 AeroSandbox / NeuralFoil / TuRBO / LLM (Qwen3.8) 的翼型元优化系统"
    )

    # 兼容直接输入参数或使用 test 子命令
    subparsers = parser.add_subparsers(dest="command", help="可选子命令: test")

    test_parser = subparsers.add_parser(
        "test", help="对指定翼型及工况进行 10 次独立对比测试评估 (Vanilla vs LLM-Meta-TuRBO)"
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
        p.add_argument(
            "--llm_url",
            type=str,
            default="http://127.0.0.1:30000/v1",
            help="SGLang / OpenAI 兼容服务地址 (默认: http://127.0.0.1:30000/v1)",
        )
        p.add_argument(
            "--llm_model",
            type=str,
            default="Qwen/Qwen3.8-27B-FP8",
            help="SGLang 部署的模型名称 (默认: Qwen/Qwen3.8-27B-FP8)",
        )
        p.add_argument(
            "--meta_steps",
            type=str,
            default="11,21,41,81,161",
            help="LLM 固定介入步数 (逗号分隔, 默认: 11,21,41,81,161)",
        )
        p.add_argument(
            "--llm_timeout",
            type=float,
            default=120.0,
            help="LLM 请求超时时间 (秒, 默认: 120.0)",
        )
        p.add_argument(
            "--api_key",
            type=str,
            default="EMPTY",
            help="SGLang / OpenAI API Key (默认: EMPTY)",
        )

    args = parser.parse_args()

    # 解析逗号分隔的固定干预步数
    try:
        meta_steps_tuple = tuple(
            int(x.strip()) for x in args.meta_steps.split(",") if x.strip()
        )
    except Exception:
        meta_steps_tuple = (11, 21, 41, 81, 161)

    run.test_turbo(
        airfoil_name=args.airfoil,
        reynolds=args.reynolds,
        alpha=args.alpha,
        n_runs=args.runs,
        llm_url=args.llm_url,
        llm_model=args.llm_model,
        meta_steps=meta_steps_tuple,
        llm_timeout=args.llm_timeout,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
