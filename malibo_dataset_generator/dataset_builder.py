"""多进程并发任务调度与元数据集构建引擎 (Dataset Builder)。

支持多核并行加速、断点增量续跑、异常重试与元数据集清单 (Manifest) 自动聚合生成。
"""

import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# 导入包内依赖
_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from malibo_dataset_generator.condition_sampler import TaskSpec
from malibo_dataset_generator.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_TASKS_DIR,
)
from malibo_dataset_generator.task_worker import execute_single_task


def _worker_wrapper(args):
    """用于 ProcessPoolExecutor 的解包包装函数"""
    task_spec, tasks_dir, force_rerun, device_str = args
    # 针对 HPC 核心配额限制 (如 10 核心): 限制单 Worker 内部线程数为 1，彻底杜绝 CPU 争抢
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)

    return execute_single_task(
        task_spec=task_spec,
        tasks_dir=tasks_dir,
        force_rerun=force_rerun,
        device_str=device_str,
    )


class DatasetBuilder:
    """高通量元优化数据集构建器"""

    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        tasks_dir: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
        max_workers: int = 8,
    ):
        self.data_dir = Path(data_dir)
        self.tasks_dir = Path(tasks_dir) if tasks_dir else self.data_dir / "tasks"
        self.manifest_path = (
            Path(manifest_path) if manifest_path else self.data_dir / "meta_dataset_manifest.json"
        )
        self.max_workers = max_workers

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def run_tasks(
        self,
        tasks: List[TaskSpec],
        force_rerun: bool = False,
        device_str: Optional[str] = None,
    ) -> Dict[str, any]:
        """多进程并发执行任务列表，聚合进度与输出统计报告。"""
        total_tasks = len(tasks)
        if total_tasks == 0:
            print("[DatasetBuilder] 没有任务需要执行。")
            return {}

        print("=" * 85)
        print(f">>> [DatasetBuilder] 开始生成 MALIBO 训练集...")
        print(f"    - 目标任务总数: {total_tasks} 个 Meta-tasks")
        print(f"    - 单任务评估步数: {tasks[0].max_evals} 步 (CMA-ES 进化优化)")
        print(f"    - 预计样本总量: {total_tasks * tasks[0].max_evals:,} 条高质量气动/几何数据")
        print(f"    - 并发工作进程数: {self.max_workers} 个 Worker 进程")
        print(f"    - 存储目录: {self.tasks_dir}")
        print("=" * 85)

        t0 = time.time()
        completed_results = []
        worker_args = [
            (spec, self.tasks_dir, force_rerun, device_str) for spec in tasks
        ]

        # 多进程并行执行
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_task = {
                executor.submit(_worker_wrapper, arg): arg[0] for arg in worker_args
            }

            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_task):
                task_spec = future_to_task[future]
                completed_count += 1
                try:
                    res = future.result()
                    completed_results.append(res)
                    status_tag = res.get("status", "DONE")
                    best_f = res.get("best_fitness", 0.0)
                    impr = res.get("improvement", 0.0)
                    elapsed = res.get("elapsed_seconds", 0.0)
                    print(
                        f"[{completed_count:3d}/{total_tasks:3d}] "
                        f"[{status_tag:9s}] {res['task_name']} "
                        f"| Best L/D: {best_f:6.2f} (Δ+{impr:5.2f}) "
                        f"| Time: {elapsed:5.1f}s"
                    )
                except Exception as exc:
                    print(
                        f"[{completed_count:3d}/{total_tasks:3d}] [ERROR] "
                        f"{task_spec.task_name}: {exc}"
                    )

        total_elapsed = time.time() - t0
        total_evals_collected = sum(
            r.get("total_evals", tasks[0].max_evals) for r in completed_results
        )

        # 构造全局 Manifest
        manifest_data = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tasks": len(completed_results),
            "target_tasks": total_tasks,
            "max_evals_per_task": tasks[0].max_evals,
            "total_evals": total_evals_collected,
            "total_elapsed_seconds": total_elapsed,
            "tasks_summary": completed_results,
        }

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)

        print("=" * 85)
        print(f">>> [DatasetBuilder] 训练集生成完成!")
        print(f"    - 成功完成任务: {len(completed_results)}/{total_tasks}")
        print(f"    - 收集评估数据条数: {total_evals_collected:,} 条")
        print(f"    - 总耗时: {total_elapsed:.1f} 秒 ({total_elapsed / 60.0:.2f} 分钟)")
        print(f"    - 数据集清单已写入: {self.manifest_path}")
        print("=" * 85)

        return manifest_data
