# AAL Brain Template with ALN Neural Mass Models - User Guide

## 概述 (Overview)

本实现提供了基于AAL（Automated Anatomical Labeling）脑模板的全脑网络模拟功能，其中每个脑区由一个ALN（Adaptive Linear-Nonlinear）神经质量模型表示。该系统支持：

This implementation provides whole-brain network simulation based on the AAL (Automated Anatomical Labeling) brain template, where each brain region is represented by an ALN (Adaptive Linear-Nonlinear) neural mass model. The system supports:

- ✅ AAL2脑模板，116个脑区（可配置）
- ✅ 每个脑区一个ALN神经质量模型（位于脑区中央）
- ✅ 可为每个ALN设置不同的参数
- ✅ 使用自定义的结构连接矩阵
- ✅ 参数优化以匹配真实功能连接数据

---

## 快速开始 (Quick Start)

### 1. 基本使用 (Basic Usage)

```python
import numpy as np
from neurolib.models.aal_aln_network import AALALNNetwork

# 创建116x116的结构连接矩阵
# Create 116x116 structural connectivity matrix
connectivity = np.random.rand(116, 116) * 0.3
connectivity = (connectivity + connectivity.T) / 2  # 对称 (symmetric)
np.fill_diagonal(connectivity, 0)  # 无自连接 (no self-connections)

# 创建网络
# Create network
network = AALALNNetwork(
    connectivity_matrix=connectivity,
    n_regions=116
)

# 运行仿真
# Run simulation
results = network.run_simulation(duration=60000)  # 60秒 (60 seconds)

# 获取结果
# Get results
firing_rates = results['firing_rates_exc']  # 兴奋性发放率
simulated_fc = results['simulated_fc']      # 仿真的功能连接
```

### 2. 加载真实功能连接数据 (Load Empirical FC Data)

```python
# 从CSV文件加载功能连接矩阵（116x116）
# Load functional connectivity matrix from CSV file (116x116)
empirical_fc = network.load_empirical_fc('your_fc_matrix.csv')

# 计算仿真FC与真实FC的相似度
# Compute similarity between simulated and empirical FC
similarity = network.compute_fc_similarity(results['simulated_fc'])
print(f"FC correlation: {similarity:.4f}")
```

### 3. 参数优化 (Parameter Optimization)

```python
# 定义参数搜索空间
# Define parameter search space
param_space = {
    'c_gl': (0.2, 0.6),          # 全局耦合强度
    'Ke_gl': (100, 400),         # 全局兴奋性连接数
    'sigmae_ext': (1.0, 2.5),    # 外部噪声
}

# 运行优化
# Run optimization
evolution = network.optimize_parameters(
    param_space=param_space,
    population_size=100,
    n_generations=50,
    output_file="optimization_results.hdf"
)

# 获取最佳参数
# Get best parameters
best_individual = evolution.dfPop.iloc[evolution.dfPop.score_0.idxmax()]
best_fitness = best_individual.score_0
print(f"Best FC correlation: {best_fitness:.4f}")
```

---

## 详细功能说明 (Detailed Features)

### 脑区特定参数 (Region-Specific Parameters)

可以为每个脑区设置不同的参数：

You can set different parameters for each brain region:

```python
# 创建每个脑区的参数列表
# Create parameter list for each region
exc_params_list = []
for i in range(116):
    params = {
        'Ke': 800.0 + np.random.randn() * 50,   # 不同的兴奋性输入
        'Ki': 200.0 + np.random.randn() * 20,   # 不同的抑制性输入
        'c_gl': 0.4 + np.random.randn() * 0.05, # 不同的耦合强度
    }
    exc_params_list.append(params)

# 创建具有脑区特定参数的网络
# Create network with region-specific parameters
network = AALALNNetwork(
    connectivity_matrix=connectivity,
    n_regions=116,
    exc_params=exc_params_list,  # 每个脑区不同的参数
)
```

### 优化脑区特定参数 (Optimize Region-Specific Parameters)

```python
# 为特定脑区定义参数空间
# Define parameter space for specific regions
param_space = {
    'c_gl': (0.2, 0.6),           # 全局参数（所有脑区相同）
    'node_0_Ke': (600, 1000),     # 脑区0的特定参数
    'node_1_Ke': (600, 1000),     # 脑区1的特定参数
    'node_2_Ki': (100, 300),      # 脑区2的特定参数
    # ... 可以为任意脑区添加特定参数
}

evolution = network.optimize_parameters(param_space=param_space, ...)
```

### 时间延迟矩阵 (Delay Matrix)

```python
# 基于距离创建延迟矩阵
# Create delay matrix based on distance
delay_matrix = distance_matrix / transmission_speed  # 毫秒 (milliseconds)

network = AALALNNetwork(
    connectivity_matrix=connectivity,
    delay_matrix=delay_matrix,
    n_regions=116
)
```

### BOLD信号仿真 (BOLD Signal Simulation)

```python
# 运行带BOLD信号的仿真（更慢但更真实）
# Run simulation with BOLD signal (slower but more realistic)
results = network.run_simulation(
    duration=60000,
    bold=True,
    chunkwise=True  # 分块计算，用于长时间仿真
)

# 使用BOLD信号进行优化
# Use BOLD signal for optimization
evolution = network.optimize_parameters(
    param_space=param_space,
    use_bold=True,
    simulation_duration=60000
)
```

---

## 完整示例脚本 (Complete Example Script)

运行提供的示例脚本：

Run the provided example script:

```bash
# 运行所有示例
# Run all examples
python examples/example_aal_aln_optimization.py

# 只运行基本仿真
# Run only basic simulation
python examples/example_aal_aln_optimization.py --example basic

# 使用自己的FC文件运行优化
# Run optimization with your own FC file
python examples/example_aal_aln_optimization.py --example optimize --fc_file your_fc.csv
```

---

## 数据格式要求 (Data Format Requirements)

### 功能连接矩阵 (Functional Connectivity Matrix)

- **格式**: CSV, NPY, 或 MAT 文件
- **维度**: 116 × 116 (或与`n_regions`匹配)
- **值范围**: -1 到 1 (相关系数)
- **属性**: 对称矩阵，对角线为1

示例CSV格式：

```csv
1.0,0.5,0.3,...
0.5,1.0,0.4,...
0.3,0.4,1.0,...
...
```

### 结构连接矩阵 (Structural Connectivity Matrix)

- **格式**: NumPy数组
- **维度**: 116 × 116
- **值范围**: 0 到 1 (归一化的连接强度)
- **属性**: 通常是对称的

### 延迟矩阵 (Delay Matrix)

- **格式**: NumPy数组
- **维度**: 116 × 116
- **值**: 毫秒 (milliseconds)
- **属性**: 对称的

---

## ALN模型参数说明 (ALN Model Parameters)

### 关键参数 (Key Parameters)

| 参数名 | 默认值 | 范围 | 说明 |
|-------|-------|------|------|
| `c_gl` | 0.4 | 0.1-0.8 | 全局耦合强度 (Global coupling strength) |
| `Ke_gl` | 250 | 100-500 | 全局兴奋性连接数 (Global excitatory connections) |
| `Ke` | 800 | 600-1000 | 局部兴奋性连接数 (Local excitatory connections) |
| `Ki` | 200 | 100-300 | 局部抑制性连接数 (Local inhibitory connections) |
| `sigmae_ext` | 1.5 | 1.0-3.0 | 外部噪声强度 (External noise strength) |
| `tau_se` | 2.0 | 1.0-5.0 | 兴奋性时间常数 (Excitatory time constant, ms) |
| `tau_si` | 5.0 | 3.0-10.0 | 抑制性时间常数 (Inhibitory time constant, ms) |
| `a` | 15.0 | 0-50 | 亚阈值适应性 (Subthreshold adaptation, nS) |
| `b` | 40.0 | 0-100 | 峰触发适应性 (Spike-triggered adaptation, pA) |

### 参数效果 (Parameter Effects)

- **`c_gl` ↑**: 增加脑区间同步 (Increases inter-regional synchrony)
- **`Ke` ↑**: 增加兴奋性，可能导致更高发放率 (Increases excitability, may lead to higher firing rates)
- **`Ki` ↑**: 增加抑制性，降低发放率 (Increases inhibition, reduces firing rates)
- **`sigmae_ext` ↑**: 增加噪声，增加变异性 (Increases noise, increases variability)
- **`a` ↑, `b` ↑**: 增加适应性，导致更多的爆发性活动 (Increases adaptation, leads to more bursting activity)

---

## 优化策略建议 (Optimization Strategy Recommendations)

### 1. 粗优化 (Coarse Optimization)

首先使用较小的种群和世代数进行快速探索：

```python
evolution = network.optimize_parameters(
    param_space=param_space,
    population_size=50,      # 较小的种群
    n_generations=20,        # 较少的世代
    simulation_duration=10000,  # 较短的仿真时间
)
```

### 2. 细优化 (Fine Optimization)

使用粗优化的结果作为起点，进行更精细的优化：

```python
# 基于粗优化结果缩小参数范围
param_space_fine = {
    'c_gl': (0.35, 0.45),  # 更窄的范围
    'Ke_gl': (200, 300),
}

evolution = network.optimize_parameters(
    param_space=param_space_fine,
    population_size=200,     # 更大的种群
    n_generations=100,       # 更多的世代
    simulation_duration=60000,  # 更长的仿真时间
    use_bold=True,          # 使用BOLD信号
)
```

### 3. 多目标优化 (Multi-Objective Optimization)

可以扩展评估函数以包含多个目标：

```python
# 在aal_aln_network.py中修改evaluate_simulation函数
# 同时优化FC相似度和FCD相似度
fitness = (fc_correlation, fcd_correlation)
```

---

## 结果分析 (Result Analysis)

### 获取脑区信息 (Get Region Information)

```python
# 获取所有脑区的信息
region_info = network.get_region_info()
print(region_info)

# 输出示例:
#    index  aal_index           name         x         y         z
# 0      0          1  Precentral_L  71.31517  133.9120  173.2864
# 1      1          2  Precentral_R  173.9577  130.6328  168.6582
# ...
```

### 保存和加载结果 (Save and Load Results)

```python
# 保存结果
network.save_results('results.npz', include_timeseries=True)

# 加载结果
data = np.load('results.npz')
connectivity = data['connectivity_matrix']
empirical_fc = data['empirical_fc']
simulated_fc = data['simulated_fc']
firing_rates = data['firing_rates_exc']
```

### 可视化 (Visualization)

使用提供的绘图函数：

```python
from examples.example_aal_aln_optimization import (
    plot_fc_comparison,
    plot_firing_rates,
    plot_optimization_progress
)

# 比较FC矩阵
plot_fc_comparison(empirical_fc, simulated_fc, save_path='fc_comparison.png')

# 绘制发放率
plot_firing_rates(firing_rates, region_names, save_path='firing_rates.png')

# 优化进度
plot_optimization_progress(evolution, save_path='optimization.png')
```

---

## 常见问题 (FAQ)

### Q1: 如何处理不同大小的脑区数量？

A: 在创建网络时指定`n_regions`参数：

```python
network = AALALNNetwork(
    connectivity_matrix=your_connectivity,  # 必须是 n×n
    n_regions=n,  # 例如 80, 90, 116, 等
)
```

### Q2: 优化需要多长时间？

A: 取决于：
- 参数空间大小
- 种群大小和世代数
- 仿真时长
- 是否使用BOLD信号
- 计算机性能

典型时间：
- 快速测试：10-30分钟
- 完整优化：数小时到数天

### Q3: 如何选择合适的参数范围？

A: 建议：
1. 从文献中的默认值开始
2. 使用宽范围进行初步探索
3. 基于初步结果缩小范围
4. 确保参数在生物学合理范围内

### Q4: 为什么优化结果不好？

可能原因：
1. **参数范围不当**：调整搜索空间
2. **仿真时间太短**：增加`simulation_duration`
3. **种群太小**：增加`population_size`
4. **世代数不足**：增加`n_generations`
5. **结构连接质量**：检查输入的连接矩阵
6. **目标不匹配**：考虑使用BOLD信号而非发放率

### Q5: 如何并行化优化？

Evolution类自动使用多核并行：

```python
# 优化会自动使用所有可用的CPU核心
evolution = network.optimize_parameters(...)
```

---

## 高级用法 (Advanced Usage)

### 自定义评估函数 (Custom Evaluation Function)

如果需要更复杂的适应度函数，可以修改`aal_aln_network.py`中的`evaluate_simulation`函数：

```python
def custom_evaluate(traj):
    model = evolution.getModelFromTraj(traj)

    # 运行仿真
    model.run()

    # 计算多个指标
    fc_score = compute_fc_correlation(model)
    fcd_score = compute_fcd_correlation(model)
    firing_rate_score = compute_firing_rate_match(model)

    # 组合多个目标
    fitness = (fc_score, fcd_score, firing_rate_score)

    return fitness, {}
```

### 使用真实DTI数据 (Using Real DTI Data)

```python
from neurolib.utils.loadData import Dataset

# 加载HCP数据集
ds = Dataset("hcp")

# 使用真实的结构连接
network = AALALNNetwork(
    connectivity_matrix=ds.Cmat,
    delay_matrix=ds.Dmat,
    n_regions=80  # HCP数据集默认80个皮层区
)

# 使用真实的功能连接
network.empirical_fc = ds.FCs[0]  # 第一个被试的FC
```

---

## 技术细节 (Technical Details)

### ALN模型架构 (ALN Model Architecture)

每个脑区包含：
- 1个兴奋性神经群体（7个状态变量）
  - 膜电流均值 (I_mu)
  - 适应电流 (I_A)
  - 兴奋性突触电流均值 (I_syn_mu_exc)
  - 抑制性突触电流均值 (I_syn_mu_inh)
  - 兴奋性突触电流方差 (I_syn_sigma_exc)
  - 抑制性突触电流方差 (I_syn_sigma_inh)
  - 平均发放率 (r_mean)

- 1个抑制性神经群体（6个状态变量）
  - 无适应电流

### 耦合方式 (Coupling Method)

- **局部耦合**: 脑区内的E-I连接
- **全局耦合**: 脑区间的兴奋性连接
- **耦合类型**: 加性耦合 (additive coupling)

### 数值积分 (Numerical Integration)

- **积分器**: jitcdde (just-in-time compiled DDE solver)
- **时间步长**: 默认0.1 ms
- **延迟处理**: 使用延迟微分方程（DDE）

---

## 引用 (Citation)

如果使用此实现，请引用：

If you use this implementation, please cite:

```bibtex
@article{cakan2020biophysically,
  title={Biophysically grounded mean-field models of neural populations under electrical stimulation},
  author={Cakan, Caglar and Obermayer, Klaus},
  journal={PLoS computational biology},
  volume={16},
  number={4},
  pages={e1007822},
  year={2020},
  publisher={Public Library of Science San Francisco, CA USA}
}

@article{roll2015implementation,
  title={Implementation of a new parcellation of the orbitofrontal cortex in the automated anatomical labeling atlas},
  author={Rolls, Edmund T and Joliot, Marc and Tzourio-Mazoyer, Nathalie},
  journal={Neuroimage},
  volume={122},
  pages={1--5},
  year={2015},
  publisher={Elsevier}
}
```

---

## 联系和支持 (Contact and Support)

- **Neurolib GitHub**: https://github.com/neurolib-dev/neurolib
- **Documentation**: https://neurolib-dev.github.io/
- **Issues**: https://github.com/neurolib-dev/neurolib/issues

---

## 更新日志 (Changelog)

### Version 1.0.0 (2025-10-23)
- ✅ 初始实现 (Initial implementation)
- ✅ AAL2模板支持 (AAL2 template support)
- ✅ 脑区特定参数 (Region-specific parameters)
- ✅ 参数优化框架 (Parameter optimization framework)
- ✅ 功能连接匹配 (Functional connectivity matching)
- ✅ 完整文档和示例 (Complete documentation and examples)

---

生成时间: 2025-10-23
由 Claude Code 生成 🤖
Generated with Claude Code 🤖
