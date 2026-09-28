"""单任务执行器与持久化存储模块 (Task Worker)。

负责为指定 TaskSpec 构建工业级气动评估器，调用 Vanilla TuRBO 运行 500 步优化，
并以原子写入方式安全导出 .npz 轨迹张量与 .json 描述文件。
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# 导入根目录评估器
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluator import AerodynamicEvaluator
from malibo_dataset_generator.condition_sampler import TaskSpec
from malibo_dataset_generator.cma_es_optimizer import (
    OptimizationTrajectory,
    run_cma_es,
)


def execute_single_task(
    task_spec: TaskSpec,
    tasks_dir: Path,
    force_rerun: bool = False,
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """执行单个优化元任务并保存完整数据集。

    支持断点续跑机制：若 .npz 文件已完整存在且 force_rerun=False，则自动跳过。
    """
    tasks_dir = Path(tasks_dir)
    tasks_dir.mkdir(parents=True, exist_ok=True)

    target_npz = tasks_dir / f"{task_spec.task_name}.npz"
    target_json = tasks_dir / f"{task_spec.task_name}.json"

    # 断点续跑检测
    if not force_rerun and target_npz.exists() and target_json.exists():
        try:
            with open(target_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("total_evals") == task_spec.max_evals:
                return {
                    "task_id": task_spec.task_id,
                    "task_name": task_spec.task_name,
                    "airfoil_name": task_spec.airfoil_name,
                    "reynolds": task_spec.reynolds,
                    "alpha": task_spec.alpha,
                    "best_fitness": meta.get("best_fitness", 0.0),
                    "initial_fitness": meta.get("initial_fitness", 0.0),
                    "improvement": meta.get("improvement", 0.0),
                    "elapsed_seconds": meta.get("elapsed_seconds", 0.0),
                    "status": "CACHED",
                    "npz_path": str(target_npz),
                }
        except Exception:
            pass  # 若读取损坏则重新运行

    # 初始化工业级评估器
    evaluator = AerodynamicEvaluator(
        reynolds=task_spec.reynolds,
        alpha=task_spec.alpha,
    )

    # 评估初始点基准
    det_init = evaluator.evaluate_detailed(task_spec.init_theta)
    init_fitness = float(det_init.get("ld_penalized", -100.0))

    import torch
    device = torch.device(device_str) if device_str else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # 启动 300 步 CMA-ES 优化 (极速且轨迹平稳均匀)
    traj = run_cma_es(
        evaluator_detailed_func=evaluator.evaluate_detailed,
        init_theta=task_spec.init_theta,
        lb=task_spec.lb,
        ub=task_spec.ub,
        task_id=task_spec.task_id,
        airfoil_name=task_spec.airfoil_name,
        reynolds=task_spec.reynolds,
        alpha=task_spec.alpha,
        max_evals=task_spec.max_evals,
        seed=task_spec.seed,
        device=device,
    )

    improvement = traj.best_fitness - init_fitness

    # 原子化保存 .npz 数组集合
    temp_npz = tasks_dir / f"{task_spec.task_name}_tmp.npz"
    np.savez_compressed(
        temp_npz,
        task_id=traj.task_id,
        airfoil_name=traj.airfoil_name,
        reynolds=traj.reynolds,
        alpha=traj.alpha,
        steps=traj.steps,
        X_unit=traj.X_unit,
        Theta=traj.Theta,
        Y_penalized=traj.Y_penalized,
        Y_raw=traj.Y_raw,
        Y_minimize=traj.Y_minimize,
        CL=traj.CL,
        CD=traj.CD,
        CM=traj.CM,
        is_feasible=traj.is_feasible,
        penalties=traj.penalties,
        best_y_history=traj.best_y_history,
        best_fitness=traj.best_fitness,
        best_theta=traj.best_theta,
    )
    os.replace(temp_npz, target_npz)

    # 保存摘要元数据 .json
    meta_dict = {
        "task_id": task_spec.task_id,
        "task_name": task_spec.task_name,
        "airfoil_name": task_spec.airfoil_name,
        "reynolds": task_spec.reynolds,
        "log_reynolds": float(np.log10(task_spec.reynolds)),
        "alpha": task_spec.alpha,
        "seed": task_spec.seed,
        "total_evals": traj.total_evals,
        "initial_fitness": init_fitness,
        "best_fitness": traj.best_fitness,
        "improvement": improvement,
        "elapsed_seconds": traj.elapsed_seconds,
        "best_theta": traj.best_theta.tolist(),
        "npz_file": str(target_npz.name),
    }

    temp_json = tasks_dir / f"{task_spec.task_name}_tmp.json"
    with open(temp_json, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2, ensure_ascii=False)
    os.replace(temp_json, target_json)

    return {
        "task_id": task_spec.task_id,
        "task_name": task_spec.task_name,
        "airfoil_name": task_spec.airfoil_name,
        "reynolds": task_spec.reynolds,
        "alpha": task_spec.alpha,
        "best_fitness": traj.best_fitness,
        "initial_fitness": init_fitness,
        "improvement": improvement,
        "elapsed_seconds": traj.elapsed_seconds,
        "status": "COMPLETED",
        "npz_path": str(target_npz),
    }
