# MALIBO 神经网络训练集自动化生成系统

本模块严格依据论文 **MALIBO: Meta-learning for Likelihood-free Bayesian Optimization** (Pan et al., 2024, arXiv:2307.03565v3)，结合工业级气动仿真评估器 `NeuralFoil ('xxxlarge')` 与贝叶斯优化算法 `Vanilla TuRBO-1`，为 MALIBO 元学习分类器（ResFFN + 任务嵌入 $z_t$）自动化生成大规模、高保真、多工况气动优化训练集。

所有新开发代码均收纳于 `malibo_dataset_generator/` 独立目录内，与根目录现有代码完全解耦。

---

## 一、 核心设计要素

### 1. 5 种迥异的起始基准翼型
为确保元模型学习到极富多样性与泛化能力的几何流场特征，精选 5 种在物理应用、厚度、弯度与弦向受载上完全迥异的基准翼型，并通过了工业级几何约束筛查（无表面相交、无鱼尾、满足厚度与前缘半径）：

| 翼型名称 | 几何构型分类 | 最大厚度 | 最大弯度 | 典型工程背景与气动流体特征 |
| :--- | :--- | :--- | :--- | :--- |
| **`NACA 0012`** | 纯对称翼型 (Symmetric) | 12.0% | 0.0% (零弯度) | 直升机旋翼、导弹舵面、平尾垂尾。上下完全对称，正负攻角流动对称。 |
| **`NACA 2412`** | 经典通航巡航型 (Conventional GA) | 12.0% | 1.9% (适度弯度) | Cessna 172、经典民用轻型机。巡航阻力桶宽广，失速平缓温和。 |
| **`NACA 6412`** | 大弯度高升力型 (High Camber) | 12.0% | 5.8% (大深弯度) | STOL 短距起降机、高空长航时无人机。极高 $C_{L0} \approx 0.7$，低速大升力。 |
| **`NACA 2418`** | 大厚度强结构型 (Thick Structural) | 18.0% | 1.8% | 大型民航内侧翼根段、风电叶根。容积与刚度大，逆压梯度剧烈。 |
| **`b737b`** | 现代喷气客机跨音速/后加载型 | 12.7% | 0.8% | 波音 737 干线客机真实中段翼型。平坦吸力面延缓激波，典型后加载修形。 |

### 2. 工况取值策略 (雷诺数与攻角)
- **雷诺数 ($Re \in [10^4, 10^8]$, 跨越 4 个量级)**：
  - 采用**常用对数均匀尺度 ($\log_{10} Re \in [4.0, 8.0]$)**；
  - 默认 5 级基准分布：$1.0\times 10^4$ (超低Re昆虫/微型翼), $1.0\times 10^5$ (低Re航模分离泡), $1.0\times 10^6$ (中Re通用通航), $1.0\times 10^7$ (高Re喷气民航), $1.0\times 10^8$ (超高Re巨型运输机/风电)。
- **攻角 ($\alpha \in [-3.0^\circ, 15.0^\circ]$)**：
  - 采用**巡航核心区间自适应加密策略**；
  - 默认 7 级基准分布：$-3.0^\circ$ (俯冲边界), $0.0^\circ$ (平飞基准), $2.5^\circ, 5.0^\circ, 7.5^\circ$ (巡航主战场加密), $11.0^\circ$ (机动爬升过渡), $15.0^\circ$ (近失速/强分离边界)。

### 3. 评估与寻优配置
- **CFD 评估器**：`NeuralFoil ('xxxlarge')`，配合 10 项航空工业级几何约束与平滑有界主导惩罚。
- **优化算法**：`CMA-ES (Covariance Matrix Adaptation Evolution Strategy)`（Hansen, 2016），严格执行单任务 300 步优化，计算极速、收敛曲线平稳线性、数据在设计空间各适应度区间分布更均匀，显著增益神经网络训练。

---

## 二、 目录与文件架构

```text
malibo_dataset_generator/
├── __init__.py
├── config.py                  # 全局超参数与工况定义 (max_evals=300, popsize=12)
├── airfoil_baselines.py       # 5 种起始翼型参数提取、边界构建与几何校验
├── condition_sampler.py       # 离散网格与连续 2D Sobol 任务采样器
├── cma_es_optimizer.py        # CMA-ES 进化优化器 (300 步平稳轨迹记录)
├── task_worker.py             # 单任务执行器与 .npz/.json 原子持久化存储
├── dataset_builder.py         # 多进程并发任务调度与全局清单 (Manifest) 生成
├── malibo_torch_dataset.py    # 适配 MALIBO 论文 (Eq. 2/3) 的 PyTorch Dataset/DataLoader
├── visualize_dataset.py       # 翼型几何、工况空间与收敛曲线可视化分析工具
├── run_generation.py          # 命令行运行入口脚本
├── README.md                  # 本使用说明文档
├── data/                      # 数据集默认输出目录 (自动创建)
│   ├── tasks/                 # 单任务 task_*.npz 与 task_*.json
│   ├── meta_dataset_manifest.json # 全局任务清单
│   └── malibo_train_dataset.pt    # PyTorch 预打包单文件
└── visualizations/            # 自动生成的分析图表 (自动创建)
```

---

## 三、 使用方法

### 1. 快速冒烟测试 (验证环境与端到端连通性)
```bash
python -m malibo_dataset_generator.run_generation --mode quick_test --workers 4
```

### 2. 生成标准全量训练集 (175 任务 x 300 步 = 52,500 条数据)
```bash
# 采用 8 个工作进程并发 (总耗时仅约 2~3 分钟，极速生成)
python -m malibo_dataset_generator.run_generation --mode grid --workers 8 --max-evals 300
```

### 3. 生成紧凑巡航训练集 (125 任务 x 300 步 = 37,500 条数据)
```bash
python -m malibo_dataset_generator.run_generation --mode compact --workers 8 --max-evals 300
```

### 4. 连续空间填充采样 (指定 T 个连续任务)
```bash
python -m malibo_dataset_generator.run_generation --mode sobol --num-tasks 100 --workers 8 --max-evals 300
```

---

## 四、 在 PyTorch 中加载与训练 MALIBO

生成的训练集完全符合 MALIBO 论文公式，并提供了极速加载支持：

```python
from malibo_dataset_generator.malibo_torch_dataset import MALIBOMetaDataset
from torch.utils.data import DataLoader

# 方式 1: 直接加载打包好的 .pt 文件 (秒级加载 87,500 条数据)
dataset = MALIBOMetaDataset(bundled_pt_path="malibo_dataset_generator/data/malibo_train_dataset.pt")

print(f"任务总数 T: {dataset.num_tasks}, 样本总数 N: {dataset.total_samples}")

# 方式 2: 获取单个任务的完整数据包
task_0 = dataset.get_task(0)
print(f"Task: {task_0.task_name} | Tau (1/3 quantile): {task_0.tau:.2f}")
print(f"X_unit: {task_0.x_unit.shape}, Labels: {task_0.labels.shape}, Weights: {task_0.weights.shape}")

# 方式 3: 抽取多任务平衡元批次 (Meta-Batch, 论文 Eq. 3)
meta_batch = dataset.sample_meta_batch(batch_size_per_task=32)
# 输出张量包含: x, task_idx, labels, weights, y
# 直接供 MALIBO 网络计算 LFBO 加权损失:
# loss = - (weights * log(C(x)) + log(1 - C(x))).mean()

# 方式 4: 使用标准 DataLoader
loader = DataLoader(dataset, batch_size=256, shuffle=True)
for x, task_idx, labels, weights, y in loader:
    # 正常训练...
    pass
```
