# AAL脑模板 + ALN神经质量模型实现

## 📋 实现概述

本实现提供了基于AAL（Automated Anatomical Labeling）脑模板的全脑网络模拟功能，其中每个脑区由一个ALN（Adaptive Linear-Nonlinear）神经质量模型表示。

### ✅ 已实现功能

1. **AAL脑模板支持**
   - 支持116个脑区（AAL2全模板，可去除小脑）
   - 每个脑区中央有一个ALN神经质量模型
   - 包含兴奋性和抑制性神经群体

2. **结构连接矩阵**
   - 支持自定义116×116结构连接矩阵
   - 支持延迟矩阵（基于纤维长度）
   - 可归一化处理

3. **灵活的参数设置**
   - 每个ALN模型可以有不同的参数
   - 支持全局参数和脑区特定参数
   - 可在优化过程中动态调整

4. **功能连接优化**
   - 加载116×116的真实功能连接矩阵（CSV格式）
   - 使用进化算法优化参数
   - 最小化模拟FC与真实FC的差异
   - 支持多目标优化

## 📁 文件结构

```
neurolib/
├── neurolib/models/aal_aln_network.py          # 主实现文件
├── examples/example_aal_aln_optimization.py     # 使用示例脚本
├── docs/AAL_ALN_NETWORK_GUIDE.md               # 详细使用指南
├── test_aal_aln.py                             # 测试脚本
└── AAL_ALN_IMPLEMENTATION_README.md            # 本文件
```

## 🚀 快速开始

### 1. 基本使用

```python
from neurolib.models.aal_aln_network import AALALNNetwork
import numpy as np

# 创建结构连接矩阵（116×116）
connectivity = np.random.rand(116, 116) * 0.3
connectivity = (connectivity + connectivity.T) / 2  # 对称化
np.fill_diagonal(connectivity, 0)  # 无自连接

# 创建网络
network = AALALNNetwork(
    connectivity_matrix=connectivity,
    n_regions=116
)

# 运行仿真
results = network.run_simulation(duration=60000)  # 60秒

# 获取结果
firing_rates = results['firing_rates_exc']
simulated_fc = results['simulated_fc']
```

### 2. 加载并匹配真实功能连接

```python
# 加载真实FC矩阵（CSV格式，116×116）
empirical_fc = network.load_empirical_fc('your_fc_matrix.csv')

# 计算相似度
similarity = network.compute_fc_similarity(results['simulated_fc'])
print(f"FC相关系数: {similarity:.4f}")
```

### 3. 参数优化

```python
# 定义参数搜索空间
param_space = {
    'c_gl': (0.2, 0.6),          # 全局耦合强度
    'Ke_gl': (100, 400),         # 全局兴奋性连接
    'sigmae_ext': (1.0, 2.5),    # 外部噪声
}

# 运行优化
evolution = network.optimize_parameters(
    param_space=param_space,
    population_size=100,
    n_generations=50,
    output_file="optimization_results.hdf"
)

# 查看最佳结果
best_individual = evolution.dfPop.iloc[evolution.dfPop.score_0.idxmax()]
print(f"最佳FC相关系数: {best_individual.score_0:.4f}")
```

### 4. 脑区特定参数

```python
# 为每个脑区设置不同的参数
exc_params_list = []
for i in range(116):
    params = {
        'Ke': 800.0 + np.random.randn() * 50,
        'Ki': 200.0 + np.random.randn() * 20,
        'c_gl': 0.4 + np.random.randn() * 0.05,
    }
    exc_params_list.append(params)

# 创建网络
network = AALALNNetwork(
    connectivity_matrix=connectivity,
    n_regions=116,
    exc_params=exc_params_list
)
```

## 📊 数据格式要求

### 功能连接矩阵（FC Matrix）

- **文件格式**: CSV, NPY, 或 MAT
- **维度**: 116 × 116
- **数值范围**: -1 到 1（相关系数）
- **属性**: 对称矩阵，对角线为1

示例CSV格式：
```csv
1.0,0.5,0.3,...
0.5,1.0,0.4,...
0.3,0.4,1.0,...
...
```

### 结构连接矩阵（Structural Connectivity）

- **格式**: NumPy数组
- **维度**: 116 × 116
- **数值范围**: 0 到 1（归一化后）
- **来源**: DTI纤维追踪

## 🔧 主要类和方法

### `AALALNNetwork` 类

主要类，用于创建和管理AAL-ALN网络。

#### 初始化参数

```python
AALALNNetwork(
    connectivity_matrix: Optional[np.ndarray] = None,
    delay_matrix: Optional[np.ndarray] = None,
    n_regions: int = 116,
    exc_params: Optional[Union[Dict, List[Dict]]] = None,
    inh_params: Optional[Union[Dict, List[Dict]]] = None,
    use_subcortical: bool = True,
)
```

#### 主要方法

| 方法 | 功能 |
|-----|------|
| `load_empirical_fc(fc_file)` | 加载真实功能连接矩阵 |
| `run_simulation(duration, dt, bold)` | 运行网络仿真 |
| `compute_fc_similarity(simulated_fc)` | 计算FC相似度 |
| `optimize_parameters(param_space, ...)` | 参数优化 |
| `update_parameters(params)` | 更新网络参数 |
| `get_region_info()` | 获取脑区信息 |
| `save_results(output_file)` | 保存结果 |

### `create_example_fc_matrix` 函数

创建示例功能连接矩阵用于测试。

```python
create_example_fc_matrix(n_regions=116, output_file="example_fc.csv")
```

## 🎯 优化策略建议

### 两阶段优化

#### 阶段1：粗优化（快速探索）

```python
evolution = network.optimize_parameters(
    param_space=param_space,
    population_size=50,
    n_generations=20,
    simulation_duration=10000,  # 10秒
)
```

#### 阶段2：细优化（精确调整）

基于阶段1的结果，缩小参数范围：

```python
# 缩小范围到最佳值附近
param_space_fine = {
    'c_gl': (best_c_gl - 0.1, best_c_gl + 0.1),
    'Ke_gl': (best_Ke_gl - 50, best_Ke_gl + 50),
}

evolution = network.optimize_parameters(
    param_space=param_space_fine,
    population_size=200,
    n_generations=100,
    simulation_duration=60000,  # 60秒
    use_bold=True,
)
```

## 🔬 关键ALN参数

| 参数 | 默认值 | 推荐范围 | 生物学意义 |
|------|--------|---------|-----------|
| `c_gl` | 0.4 | 0.1-0.8 | 全局耦合强度，控制脑区间同步 |
| `Ke_gl` | 250 | 100-500 | 全局兴奋性连接数 |
| `Ke` | 800 | 600-1000 | 局部兴奋性连接数 |
| `Ki` | 200 | 100-300 | 局部抑制性连接数 |
| `sigmae_ext` | 1.5 | 1.0-3.0 | 外部噪声强度 |
| `tau_se` | 2.0 | 1.0-5.0 | 兴奋性时间常数（毫秒） |
| `tau_si` | 5.0 | 3.0-10.0 | 抑制性时间常数（毫秒） |
| `a` | 15.0 | 0-50 | 亚阈值适应性（nS） |
| `b` | 40.0 | 0-100 | 峰触发适应性（pA） |

## 📈 示例脚本

### 运行完整示例

```bash
# 运行所有示例
python examples/example_aal_aln_optimization.py

# 只运行基本仿真
python examples/example_aal_aln_optimization.py --example basic

# 使用自己的FC文件
python examples/example_aal_aln_optimization.py --example optimize --fc_file your_fc.csv
```

### 示例包含：

1. **基本仿真**: 创建网络并运行简单仿真
2. **脑区特定参数**: 为不同脑区设置不同参数
3. **参数优化**: 完整的优化流程示例

## ⚠️ 注意事项

### 计算资源

- **内存**: 建议至少8GB RAM
- **CPU**: 优化会自动使用多核并行
- **时间**: 完整优化可能需要数小时到数天

### 参数选择

1. **从默认值开始**: 使用文献中的默认参数
2. **逐步调整**: 先调整全局参数，再调整局部参数
3. **验证合理性**: 确保参数在生物学合理范围内
4. **检查活动**: 确保神经活动在合理范围（1-100 Hz）

### 常见问题

#### Q: 仿真结果不稳定？
A: 尝试：
- 增加仿真时长
- 调整噪声水平
- 检查初始条件

#### Q: 优化收敛慢？
A: 尝试：
- 增加种群大小
- 扩大参数搜索范围
- 使用更多世代

#### Q: FC相关系数低？
A: 可能原因：
- 结构连接质量不好
- 参数范围不合适
- 仿真时长太短
- 考虑使用BOLD信号而非发放率

## 📚 技术细节

### ALN模型

- **基于**: 适应性指数积分发放（AdEx）神经元
- **方法**: Fokker-Planck均场近似
- **查找表**: 预计算的传递函数（h5格式）
- **积分器**: jitcdde（just-in-time DDE求解器）

### 优化算法

- **算法**: NSGA-II（多目标进化算法）
- **库**: DEAP（Distributed Evolutionary Algorithms in Python）
- **并行**: 自动多核并行化
- **存储**: HDF5格式（通过pypet）

### 网络结构

- 每个脑区 = 1个ALN节点
- 每个节点 = 1个兴奋性 + 1个抑制性神经群
- 兴奋性群体: 7个状态变量
- 抑制性群体: 6个状态变量
- 总状态变量数 = 116 × (7 + 6) = 1508

## 📖 参考文献

1. **ALN模型**:
   ```
   Cakan, C., & Obermayer, K. (2020).
   Biophysically grounded mean-field models of neural populations
   under electrical stimulation.
   PLoS Computational Biology, 16(4), e1007822.
   ```

2. **AAL2模板**:
   ```
   Rolls, E. T., Joliot, M., & Tzourio-Mazoyer, N. (2015).
   Implementation of a new parcellation of the orbitofrontal cortex
   in the automated anatomical labeling atlas.
   NeuroImage, 122, 1-5.
   ```

3. **Neurolib**:
   ```
   https://github.com/neurolib-dev/neurolib
   ```

## 🛠️ 测试

运行测试脚本验证安装：

```bash
python test_aal_aln.py
```

测试包括：
- ✓ 基本网络创建
- ✓ FC矩阵加载
- ✓ 脑区特定参数
- ✓ 仿真运行

## 📝 完整文档

详细使用指南请参阅：
- `docs/AAL_ALN_NETWORK_GUIDE.md` - 完整中英文使用指南

## 💻 系统要求

- Python 3.7+
- NumPy
- SciPy
- Pandas
- h5py
- numba
- jitcdde
- deap
- pypet
- 其他依赖见 `requirements.txt`

## 🤝 贡献

这是使用Claude Code自动生成的实现。如有问题或改进建议，请：
1. 检查文档是否已解决问题
2. 运行测试脚本定位问题
3. 提交详细的问题描述

## 📧 联系方式

- Neurolib GitHub: https://github.com/neurolib-dev/neurolib
- Neurolib文档: https://neurolib-dev.github.io/

---

**生成时间**: 2025-10-23
**生成工具**: 🤖 Claude Code
**版本**: 1.0.0
