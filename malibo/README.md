# MALIBO: Meta-learning for Likelihood-free Bayesian Optimization

本模块根据论文 **MALIBO: Meta-learning for Likelihood-free Bayesian Optimization** (Pan et al., 2024, arXiv:2307.03565v3) 完整实现，并严格遵循 Python Vibe Coding 最佳实践（强类型标注、软间隔全数值稳定数学公式、无静默异常吞没、模块解耦）。

所有代码均收纳在 `malibo/` 独立文件夹中，与根目录已有代码完全隔离。

---

## 一、 算法核心与论文数学映射

| 论文公式/章节 | 模块组件 | 数学逻辑与实现细节 |
| :--- | :--- | :--- |
| **Section 4 (Figure 2)** | [`models.py`](file:///home/msds/c250089/notebook/CFD/malibo/models.py) | **ResFFN 骨干与双重预测层**：4 个残差块（每块 64 单元，ELU 激活）映射原始参数至 $\Phi \in \mathbb{R}^{50}$；任务无关均值层 $m(\Phi) \to \mathbb{R}$；各任务嵌入向量 $z_t \in \mathbb{R}^{50}$，输出 Logit $a = m(\Phi) + z_t^T \Phi$。 |
| **Eq. (2) & Eq. (6)** | [`losses.py`](file:///home/msds/c250089/notebook/CFD/malibo/losses.py) | **LFBO 效用加权损失**：基于分位数 $\gamma=1/3$ 划分阈值 $\tau$。推导使用软间隔连续形式：$\mathcal{L} = w \cdot \text{softplus}(-a) + \text{softplus}(a)$，彻底杜绝 $\log(0)$、NaN 与除零错误。 |
| **Eq. (4) & Appendix C** | [`losses.py`](file:///home/msds/c250089/notebook/CFD/malibo/losses.py) | **潜在任务空间高斯正则化器 $R(Z)$**：结合 Kolmogorov-Smirnov (KS) 边际分布匹配与经验协方差对齐，自动根据先验样本估算 $\lambda_{KS}$ 与 $\lambda_{Cov}$ 归一化系数，约束 $z_t \sim \mathcal{N}(0, I)$。 |
| **Eq. (6) ~ Eq. (9)** | [`adaptation.py`](file:///home/msds/c250089/notebook/CFD/malibo/adaptation.py) | **任务自适应与 Laplace 近似**：固定特征网络，在目标任务上通过 L-BFGS（`strong_wolfe`）求解 $z_{MAP}$；计算精确负 Hessian 精度矩阵 $\Sigma_N^{-1}$ 并进行 Cholesky 分解，抽取 Thompson 样本 $\hat{z} \sim \mathcal{N}(z_{MAP}, \Sigma_N)$。 |
| **Eq. (10) & Appendix G** | [`residual_booster.py`](file:///home/msds/c250089/notebook/CFD/malibo/residual_booster.py) | **决策树残差梯度提升 (GBDT)**：以 Meta-Model 的 Thompson 采样为基分类器（`InitEstimator`），通过 70% 训练 / 30% 验证早停确定树数量，叠加误差残差修正 $C_{GB}(x) = \sigma(m(\phi(x)) + \hat{z}^T \phi(x) + \sum r_i(x))$。 |
| **Algorithm 1** | [`optimizer.py`](file:///home/msds/c250089/notebook/CFD/malibo/optimizer.py) | **完整元学习贝叶斯优化闭环**：第 0 步由 $g_\omega(x)$ 推荐 $x_0$；后续步自适应更新 $z$ 并融入残差树，通过 5,120 个低差异 Sobol 候选点极大化采集函数。 |

---

## 二、 文件组织清单

```text
malibo/
├── __init__.py            # 导出顶级模型、训练器与优化器接口
├── config.py              # 对齐论文 Appendix G 的官方默认超参数
├── models.py              # ResFFN, MeanHead, MALIBOMetaModel
├── losses.py              # LFBOLoss, LatentSpaceRegularizer
├── adaptation.py          # L-BFGS MAP, Laplace Cholesky, Thompson Sampling
├── residual_booster.py    # scikit-learn GradientBoosting 强基学习器与残差叠加
├── trainer.py             # Adam + LR 衰减 + 正则化多任务元训练循环
├── optimizer.py           # Algorithm 1 优化器 (ask/tell 范式)
├── train_malibo.py        # 训练命令行入口
├── test_airfoil_bo.py     # 气动优化测试与 Vanilla TuRBO 对比评测脚本
└── README.md              # 架构与使用说明
```

---

## 三、 使用指南

### 1. 训练 MALIBO 元神经网络
使用此前生成的 87,500 条完整气动数据集进行全量元训练（在 L40S GPU 上训练极其迅速）：

```bash
# 全量训练 2048 轮 (带早停，权重自动保存至 malibo/malibo_checkpoint.pt)
python -m malibo.train_malibo \
    --dataset malibo_dataset_generator/data/malibo_train_dataset.pt \
    --epochs 2048 \
    --lr 1e-3 \
    --batch-size 256 \
    --output malibo/malibo_checkpoint.pt
```

### 2. 测试气动优化效果（对比 Vanilla TuRBO）
在全新的未见过工况下测试训练好的 MALIBO 优化能力，并与 Vanilla TuRBO-1 进行同条件对比：

```bash
# 在 NACA 2412 @ Re=2.5e6, Alpha=5.0° 工况下测试 50 步对比
python -m malibo.test_airfoil_bo \
    --checkpoint malibo/malibo_checkpoint.pt \
    --airfoil naca2412 \
    --reynolds 2.5e6 \
    --alpha 5.0 \
    --budget 50 \
    --compare-turbo \
    --output-plot malibo/malibo_vs_turbo.png
```

---

## 四、 Python 代码调用范例

您也可以在自己的 Python 脚本中按组件模块化调用：

```python
import torch
from malibo import MALIBOMetaModel, MALIBOOptimizer
from evaluator import AerodynamicEvaluator

# 1. 加载训练好的 MALIBO 检查点
ckpt = torch.load("malibo/malibo_checkpoint.pt", map_location="cuda", weights_only=False)
model = MALIBOMetaModel(num_tasks=ckpt["num_tasks"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval().cuda()

# 2. 创建优化器实例
opt = MALIBOOptimizer(
    meta_model=model,
    dim=18,
    lb=lb,
    ub=ub,
    device=torch.device("cuda"),
)

# 3. 运行闭环优化
evaluator = AerodynamicEvaluator(reynolds=1e6, alpha=4.0)
results = opt.optimize(objective_func=evaluator.evaluate, budget=100)

print(f"MALIBO 找到的最佳升阻比: {results['best_y']:.2f}")
```
