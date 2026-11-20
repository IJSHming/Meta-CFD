# Meta-CFD: 2D Airfoil Aerodynamic Evaluator & Paper-Exact MALIBO Framework

本仓库提供面向 2D 翼型（18 维 Kulfan CST 参数化）的工业级几何/气动评估器、离线多工况元学习轨迹生成管道，以及**严格对齐原论文 Pan et al. (ICML 2024, `MALIBO.pdf`) 的无似然元贝叶斯优化（MALIBO）核心实现**（供 `RL-Airfoil` 与本仓库测试直接复用）。

## 核心模块

- `evaluator.py` & `test_constraints.py`：基于 `AeroSandbox` + `NeuralFoil ('xxxlarge')` 的 10 项工业级强约束翼型评估器（最大厚度 `[8%, 20%]`、结构梁最小厚度、后缘防鱼尾单调收缩、前缘钝度半径、后缘楔角、截面面积、抗波纹凸性、俯仰力矩 $C_M \ge -0.25$）及 5 项约束单元测试。
- `malibo/`（详见 [malibo/README.md](malibo/README.md)）：严格忠实于 Pan et al. (ICML 2024) 原文的 MALIBO 核心实现——`4×64` ResFFN（ELU, $d=50$）、精确 Eq. (2) LFBO 效用损失、精确非参数排序 KS + 协方差匹配高斯先验正则（元训练后 $z_t$ 收敛至 $\mathrm{std}=0.9980, |\mathrm{mean}|=0.0029$）、L-BFGS 拉普拉斯汤普森采样（固定 $\gamma=1/3$）、70/30 早停双标签残差 GBDT 及原论文 Appendix G 规定的全局 5,120 点均匀随机搜索（无信赖域、无稀疏坐标掩码、无免费几何预筛器）。
- `malibo_dataset_generator/`（详见 [malibo_dataset_generator/README.md](malibo_dataset_generator/README.md)）：基于 18 维统一 Kulfan 搜索盒（与 `RL-Airfoil` 严格对齐）及多进程 CMA-ES（`spawn` 模式）的离线多任务数据集生成器，已构建 **250 任务 $\times$ 300 步 = 75,000 样本**元训练集（5 翼型 $\times$ 10 雷诺数 $\times$ 5 攻角，严格排除测试工况 $Re=3.5\times 10^6, \alpha=3.5^\circ$）。
- `turbo.py`, `llm_agent.py`, `run.py`, `main.py`：早期 BoTorch TuRBO-1 与 SGLang LLM 交互原型脚本（主论文 10 种子全量 2D 基准测试详见 `RL-Airfoil/` 仓库）。

## 快速验证与复现

```bash
# 1. 运行 5 项工业级几何约束单元测试
python test_constraints.py

# 2. 重新生成 250 任务 x 300 步 (75,000 样本) 元训练集
python -m malibo_dataset_generator.run_generation --mode compact --workers 10 --max-evals 300

# 3. 训练严格原论文对齐版 MALIBO 权重 (2048 epochs + 80/20 早停)
python -m malibo.train_malibo --dataset malibo_dataset_generator/data/malibo_train_dataset.pt

# 4. 在未见测试工况下运行 MALIBO vs Vanilla TuRBO 对比测试
python -m malibo.test_airfoil_bo --airfoil naca2412 --reynolds 2.5e6 --alpha 5.0 --budget 100 --compare-turbo
```
