"""
Improved AAL Brain Template with ALN Neural Mass Models

这个改进版本修复了以下问题：
1. 使用ALNModel而不是ALNNetwork（更简单直接）
2. 简化了参数处理逻辑
3. 改进了错误处理和日志
4. 优化了参数映射机制
5. 添加了参数验证

Author: Claude Code (Improved Version)
Date: 2025-10-25
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple

from neurolib.models.aln import ALNModel
from neurolib.utils.atlases import AutomatedAnatomicalParcellation2
from neurolib.optimize.evolution import Evolution
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.utils import functions as func


class AALALNNetwork:
    """
    基于AAL脑模板的全脑ALN网络

    简化版实现：
    - 使用ALNModel作为底层模型（支持全脑网络）
    - 每个脑区由一个节点表示
    - 支持参数优化以匹配功能连接

    注意：ALNModel本身已经包含了多个节点（每个脑区一个节点），
    每个节点包含兴奋性和抑制性神经群体。
    """

    # ALNModel参数名称映射（用户友好名称 -> ALNModel内部名称）
    PARAM_ALIASES = {
        'c_gl': 'K_gl',              # 全局耦合强度
        'Ke_gl': 'Ke_gl',            # 全局兴奋性连接
        'mue_ext': 'mue_ext_mean',   # 兴奋性外部输入
        'mui_ext': 'mui_ext_mean',   # 抑制性外部输入
        'noise': 'sigma_ou',         # OU噪声强度
    }

    def __init__(
        self,
        connectivity_matrix: Optional[np.ndarray] = None,
        delay_matrix: Optional[np.ndarray] = None,
        n_regions: int = 116,
        use_subcortical: bool = True,
        **model_params
    ):
        """
        初始化AAL-ALN网络

        参数：
            connectivity_matrix: 结构连接矩阵 (n_regions x n_regions)
            delay_matrix: 延迟矩阵，单位毫秒 (n_regions x n_regions)
            n_regions: 脑区数量（默认116）
            use_subcortical: 是否包含皮层下区域
            **model_params: 传递给ALNModel的额外参数
        """
        self.n_regions = n_regions
        self.use_subcortical = use_subcortical

        # 初始化AAL2脑图谱
        logging.info("初始化AAL2脑图谱...")
        self.atlas = AutomatedAnatomicalParcellation2()

        # 选择脑区
        self._select_brain_regions()

        # 准备连接矩阵
        self.connectivity_matrix = self._prepare_connectivity(connectivity_matrix)
        self.delay_matrix = self._prepare_delays(delay_matrix)

        # 创建ALN模型
        logging.info("创建ALN全脑模型...")
        self.model = ALNModel(
            Cmat=self.connectivity_matrix,
            Dmat=self.delay_matrix,
            **model_params
        )

        # 设置默认参数
        self._set_default_parameters()

        # 存储实验数据
        self.empirical_fc = None

        logging.info(f"✓ AAL-ALN网络初始化完成！({self.n_regions}个脑区)")

    def _select_brain_regions(self):
        """选择要使用的脑区"""
        if self.use_subcortical:
            # 使用前n_regions个区域（包含皮层下）
            available = min(self.n_regions, len(self.atlas))
            self.region_indices = list(range(available))
        else:
            # 只使用皮层区域
            self.region_indices = self.atlas.cortex[:self.n_regions]

        self.n_regions = len(self.region_indices)
        self.region_names = [self.atlas[i] for i in self.region_indices]

        logging.info(f"选择了 {self.n_regions} 个脑区")
        logging.info(f"脑区示例: {self.region_names[:3]}")

        # 获取脑区坐标
        self._get_region_coordinates()

    def _get_region_coordinates(self):
        """获取脑区的空间坐标"""
        try:
            # AAL2的坐标存储在aal2_centers中
            coords_array = self.atlas.aal2_centers  # shape: (3, 120)

            # 转置并选择我们需要的区域
            self.region_coords = []
            for idx in self.region_indices:
                if idx < coords_array.shape[1]:
                    # 获取该区域的x, y, z坐标
                    coord = [
                        float(coords_array[0, idx]),  # x
                        float(coords_array[1, idx]),  # y
                        float(coords_array[2, idx])   # z
                    ]
                    self.region_coords.append(coord)
                else:
                    self.region_coords.append([0.0, 0.0, 0.0])

            logging.info(f"成功获取 {len(self.region_coords)} 个脑区坐标")

        except Exception as e:
            logging.warning(f"无法获取脑区坐标: {e}")
            self.region_coords = None

    def _prepare_connectivity(self, connectivity_matrix):
        """准备结构连接矩阵"""
        if connectivity_matrix is None:
            logging.warning("未提供连接矩阵，使用单位矩阵")
            return np.eye(self.n_regions) * 0.1

        # 验证形状
        if connectivity_matrix.shape != (self.n_regions, self.n_regions):
            raise ValueError(
                f"连接矩阵形状 {connectivity_matrix.shape} "
                f"与脑区数量 {self.n_regions} 不匹配"
            )

        # 确保对称性
        if not np.allclose(connectivity_matrix, connectivity_matrix.T):
            logging.warning("连接矩阵不对称，将进行对称化处理")
            connectivity_matrix = (connectivity_matrix + connectivity_matrix.T) / 2

        return connectivity_matrix

    def _prepare_delays(self, delay_matrix):
        """准备延迟矩阵"""
        if delay_matrix is None:
            logging.info("未提供延迟矩阵，使用零延迟")
            return np.zeros((self.n_regions, self.n_regions))

        if delay_matrix.shape != (self.n_regions, self.n_regions):
            raise ValueError(
                f"延迟矩阵形状 {delay_matrix.shape} "
                f"与脑区数量 {self.n_regions} 不匹配"
            )

        return delay_matrix

    def _set_default_parameters(self):
        """设置默认的模型参数"""
        # 只设置关键参数，其他使用ALNModel默认值
        defaults = {
            'duration': 60000,      # 60秒
            'dt': 0.1,             # 0.1毫秒时间步长
            'sigma_ou': 0.1,       # 较小的噪声
        }

        for key, value in defaults.items():
            if key not in self.model.params:
                self.model.params[key] = value

    def load_empirical_fc(
        self,
        fc_file: str,
        file_format: str = 'csv'
    ) -> np.ndarray:
        """
        加载实验功能连接矩阵

        参数：
            fc_file: FC矩阵文件路径
            file_format: 文件格式 ('csv', 'npy', 'mat')

        返回：
            功能连接矩阵 (n_regions x n_regions)
        """
        logging.info(f"从 {fc_file} 加载实验FC矩阵...")

        # 加载文件
        if file_format == 'csv':
            fc_matrix = pd.read_csv(fc_file, header=None).values
        elif file_format == 'npy':
            fc_matrix = np.load(fc_file)
        elif file_format == 'mat':
            import scipy.io
            mat_data = scipy.io.loadmat(fc_file)
            # 尝试常见的键名
            for key in ['fc', 'FC', 'correlation', 'corr', 'data']:
                if key in mat_data:
                    fc_matrix = mat_data[key]
                    break
            else:
                raise ValueError(
                    f"在.mat文件中找不到FC矩阵。"
                    f"可用键: {list(mat_data.keys())}"
                )
        else:
            raise ValueError(f"不支持的文件格式: {file_format}")

        # 验证形状
        if fc_matrix.shape != (self.n_regions, self.n_regions):
            raise ValueError(
                f"FC矩阵形状 {fc_matrix.shape} "
                f"与脑区数量 {self.n_regions} 不匹配"
            )

        # 数据清理
        fc_matrix = np.nan_to_num(fc_matrix)  # 移除NaN
        fc_matrix = (fc_matrix + fc_matrix.T) / 2  # 对称化
        np.fill_diagonal(fc_matrix, 1.0)  # 对角线设为1

        # 验证值范围
        if fc_matrix.min() < -1 or fc_matrix.max() > 1:
            logging.warning(
                f"FC矩阵值超出[-1,1]范围: "
                f"[{fc_matrix.min():.3f}, {fc_matrix.max():.3f}]"
            )

        self.empirical_fc = fc_matrix
        logging.info(f"✓ 成功加载FC矩阵: {fc_matrix.shape}")

        return fc_matrix

    def run_simulation(
        self,
        duration: Optional[float] = None,
        dt: Optional[float] = None,
        bold: bool = False,
        chunkwise: bool = False,
    ) -> Dict:
        """
        运行网络仿真

        参数：
            duration: 仿真时长（毫秒），None使用模型默认值
            dt: 时间步长（毫秒），None使用模型默认值
            bold: 是否计算BOLD信号
            chunkwise: 是否分块计算（用于长时间仿真）

        返回：
            包含仿真结果的字典
        """
        # 设置参数
        if duration is not None:
            self.model.params['duration'] = duration
        if dt is not None:
            self.model.params['dt'] = dt

        logging.info(
            f"运行仿真: {self.model.params['duration']}ms, "
            f"dt={self.model.params['dt']}ms"
        )

        # 运行仿真
        try:
            if chunkwise:
                self.model.run(chunkwise=True, bold=bold)
            else:
                self.model.run(bold=bold)
        except Exception as e:
            logging.error(f"仿真失败: {e}")
            raise

        # 提取结果
        results = {
            'model': self.model,
            'time': self.model.t if hasattr(self.model, 't') else None,
        }

        # 提取发放率
        if hasattr(self.model, 'output'):
            results['firing_rates'] = self.model.output

        # 计算功能连接
        if bold and hasattr(self.model, 'BOLD'):
            bold_signal = self.model.BOLD.BOLD
            # 跳过初始瞬态
            skip_samples = min(5, bold_signal.shape[1] // 10)
            results['bold_signal'] = bold_signal
            results['simulated_fc'] = func.fc(bold_signal[:, skip_samples:])
            logging.info("使用BOLD信号计算FC")
        elif hasattr(self.model, 'output'):
            results['simulated_fc'] = func.fc(self.model.output)
            logging.info("使用发放率计算FC")
        else:
            logging.warning("无法计算FC：没有可用的输出")
            results['simulated_fc'] = None

        logging.info("✓ 仿真完成")

        return results

    def compute_fc_similarity(self, simulated_fc: np.ndarray) -> float:
        """
        计算仿真FC与实验FC的相似度

        参数：
            simulated_fc: 仿真得到的功能连接矩阵

        返回：
            Pearson相关系数
        """
        if self.empirical_fc is None:
            raise ValueError("请先使用 load_empirical_fc() 加载实验数据")

        correlation = func.matrix_correlation(simulated_fc, self.empirical_fc)

        return correlation

    def optimize_parameters(
        self,
        param_space: Dict[str, Tuple[float, float]],
        population_size: int = 100,
        n_generations: int = 50,
        initial_population_size: Optional[int] = None,
        output_file: str = "aal_aln_optimization.hdf",
        use_bold: bool = False,
        simulation_duration: float = 30000,
        quick_check_duration: float = 3000,
        verbose: bool = True,
    ) -> Evolution:
        """
        优化参数以匹配实验功能连接

        参数：
            param_space: 参数搜索空间字典
                        键：参数名（可使用别名，如'c_gl'）
                        值：(最小值, 最大值) 元组
            population_size: 每代个体数量
            n_generations: 进化代数
            initial_population_size: 初始种群大小（默认为population_size*4）
            output_file: 保存结果的HDF5文件
            use_bold: 是否使用BOLD信号（更慢但更真实）
            simulation_duration: 完整仿真时长（毫秒）
            quick_check_duration: 快速检查仿真时长（毫秒）
            verbose: 是否打印详细信息

        返回：
            Evolution对象
        """
        if self.empirical_fc is None:
            raise ValueError("请先使用 load_empirical_fc() 加载实验数据")

        if initial_population_size is None:
            initial_population_size = population_size * 4

        logging.info("="*60)
        logging.info("开始参数优化")
        logging.info("="*60)
        logging.info(f"参数空间: {param_space}")
        logging.info(f"种群大小: {population_size}")
        logging.info(f"进化代数: {n_generations}")
        logging.info(f"使用BOLD: {use_bold}")

        # 映射参数别名到实际参数名
        mapped_param_space = {}
        for user_name, bounds in param_space.items():
            actual_name = self.PARAM_ALIASES.get(user_name, user_name)
            mapped_param_space[actual_name] = bounds

        # 创建参数空间对象
        param_names = list(mapped_param_space.keys())
        param_ranges = [mapped_param_space[name] for name in param_names]
        pspace = ParameterSpace(param_names, param_ranges)

        # 定义适应度评估函数
        def evaluate_simulation(traj):
            """评估一组参数的适应度"""
            try:
                # 从轨迹中提取参数
                ind_dict = pspace.dict_from_trajectory(traj)

                # 更新模型参数
                model = self.model
                original_params = {}

                for param_name, param_value in ind_dict.items():
                    original_params[param_name] = model.params.get(param_name)
                    model.params[param_name] = param_value

                # 阶段1: 快速检查活动是否合理
                model.params['duration'] = quick_check_duration
                model.run()

                # 检查输出
                if not hasattr(model, 'output') or model.output is None:
                    return (0.0,), {}

                # 检查发放率范围
                output = model.output
                t_check = model.t > (quick_check_duration / 2)

                try:
                    max_rate = np.max(output[:, t_check])
                    mean_rate = np.mean(output[:, t_check])
                except:
                    max_rate = np.max(output)
                    mean_rate = np.mean(output)

                # 活动过高或过低
                if max_rate > 200 or max_rate < 0.01:
                    logging.debug(
                        f"活动异常: max={max_rate:.2f}, mean={mean_rate:.2f}"
                    )
                    return (0.0,), {}

                # 阶段2: 完整仿真
                model.params['duration'] = simulation_duration

                if use_bold:
                    model.run(chunkwise=True, bold=True)
                    if hasattr(model, 'BOLD') and hasattr(model.BOLD, 'BOLD'):
                        bold_signal = model.BOLD.BOLD
                        skip = min(5, bold_signal.shape[1] // 10)
                        simulated_fc = func.fc(bold_signal[:, skip:])
                    else:
                        return (0.0,), {}
                else:
                    model.run()
                    if hasattr(model, 'output'):
                        simulated_fc = func.fc(model.output)
                    else:
                        return (0.0,), {}

                # 计算适应度
                fitness = func.matrix_correlation(simulated_fc, self.empirical_fc)

                if np.isnan(fitness) or np.isinf(fitness):
                    fitness = 0.0

                # 恢复原参数
                for param_name, original_value in original_params.items():
                    if original_value is not None:
                        model.params[param_name] = original_value

                return (fitness,), {}

            except Exception as e:
                logging.debug(f"评估失败: {e}")
                return (0.0,), {}

        # 创建Evolution对象
        evolution = Evolution(
            evaluate_simulation,
            pspace,
            algorithm='nsga2',
            weightList=[1.0],  # 最大化FC相关性
            model=self.model,
            POP_INIT_SIZE=initial_population_size,
            POP_SIZE=population_size,
            NGEN=n_generations,
            filename=output_file,
        )

        # 运行优化
        logging.info("\n开始进化优化...")
        evolution.run(verbose=verbose)

        # 获取最佳结果
        if hasattr(evolution, 'dfPop') and len(evolution.dfPop) > 0:
            best_idx = evolution.dfPop.score_0.idxmax()
            best_individual = evolution.dfPop.iloc[best_idx]
            best_fitness = best_individual.score_0

            logging.info("\n" + "="*60)
            logging.info("优化完成！")
            logging.info("="*60)
            logging.info(f"最佳适应度 (FC相关性): {best_fitness:.4f}")
            logging.info("最佳参数:")
            for param_name in param_names:
                value = best_individual[param_name]
                logging.info(f"  {param_name}: {value:.4f}")
        else:
            logging.warning("优化完成，但没有有效结果")

        return evolution

    def update_parameters(self, params: Dict[str, float]):
        """
        更新模型参数

        参数：
            params: 参数字典（可使用别名）
        """
        for user_name, value in params.items():
            # 映射别名到实际参数名
            actual_name = self.PARAM_ALIASES.get(user_name, user_name)
            self.model.params[actual_name] = value
            logging.info(f"更新参数: {user_name} ({actual_name}) = {value:.4f}")

    def get_region_info(self) -> pd.DataFrame:
        """
        获取所有脑区的信息

        返回：
            包含脑区信息的DataFrame
        """
        info_list = []

        for i, (idx, name) in enumerate(zip(self.region_indices, self.region_names)):
            info_dict = {
                'index': i,
                'aal_index': idx,
                'name': name,
            }

            # 添加坐标信息
            if self.region_coords is not None and i < len(self.region_coords):
                info_dict['x'] = self.region_coords[i][0]
                info_dict['y'] = self.region_coords[i][1]
                info_dict['z'] = self.region_coords[i][2]

            info_list.append(info_dict)

        return pd.DataFrame(info_list)

    def save_results(
        self,
        output_file: str,
        include_timeseries: bool = False
    ):
        """
        保存网络结果

        参数：
            output_file: 输出文件路径（.npz格式）
            include_timeseries: 是否保存完整时间序列
        """
        results = {
            'connectivity_matrix': self.connectivity_matrix,
            'delay_matrix': self.delay_matrix,
            'region_names': np.array(self.region_names, dtype=object),
            'region_indices': np.array(self.region_indices),
            'n_regions': self.n_regions,
        }

        if self.empirical_fc is not None:
            results['empirical_fc'] = self.empirical_fc

        if hasattr(self.model, 'output') and self.model.output is not None:
            results['simulated_fc'] = func.fc(self.model.output)

            if include_timeseries:
                results['firing_rates'] = self.model.output
                results['time'] = self.model.t if hasattr(self.model, 't') else None

        if self.region_coords is not None:
            results['region_coordinates'] = np.array(self.region_coords)

        np.savez_compressed(output_file, **results)
        logging.info(f"✓ 结果已保存到: {output_file}")

    def get_model_params(self) -> Dict:
        """获取当前模型参数"""
        return dict(self.model.params)

    def print_summary(self):
        """打印网络摘要信息"""
        print("\n" + "="*60)
        print("AAL-ALN Network Summary")
        print("="*60)
        print(f"脑区数量: {self.n_regions}")
        print(f"包含皮层下: {self.use_subcortical}")
        print(f"连接矩阵形状: {self.connectivity_matrix.shape}")
        print(f"延迟矩阵形状: {self.delay_matrix.shape}")
        print(f"实验FC已加载: {'是' if self.empirical_fc is not None else '否'}")
        print("\n关键参数:")
        key_params = ['K_gl', 'Ke_gl', 'mue_ext_mean', 'mui_ext_mean',
                     'sigma_ou', 'duration', 'dt']
        for param in key_params:
            if param in self.model.params:
                print(f"  {param}: {self.model.params[param]}")
        print("="*60 + "\n")


def create_example_fc_matrix(
    n_regions: int = 116,
    output_file: str = "example_fc_116x116.csv",
    spatial_decay: float = 10.0,
    noise_level: float = 0.3,
) -> np.ndarray:
    """
    创建示例功能连接矩阵

    生成具有真实特性的合成FC矩阵：
    - 对称矩阵
    - 对角线为1
    - 相邻脑区有更高相关性（空间衰减）
    - 添加适量噪声

    参数：
        n_regions: 脑区数量
        output_file: 输出CSV文件路径
        spatial_decay: 空间衰减参数（越大衰减越慢）
        noise_level: 噪声水平

    返回：
        功能连接矩阵 (n_regions x n_regions)
    """
    logging.info(f"创建示例FC矩阵 ({n_regions}x{n_regions})...")

    np.random.seed(42)  # 可重复性

    # 生成基础随机矩阵
    fc = np.random.randn(n_regions, n_regions) * noise_level
    fc = (fc + fc.T) / 2  # 对称化

    # 添加空间结构（相邻脑区更相关）
    for i in range(n_regions):
        for j in range(n_regions):
            distance = abs(i - j)
            # 添加距离依赖的相关性
            spatial_corr = 2.0 * np.exp(-distance / spatial_decay)
            fc[i, j] += spatial_corr

    # 归一化为相关矩阵
    D = np.sqrt(np.diag(fc))
    D[D == 0] = 1  # 避免除零
    fc = fc / np.outer(D, D)

    # 确保值在[-1, 1]范围内
    fc = np.clip(fc, -1, 1)

    # 对角线设为1
    np.fill_diagonal(fc, 1.0)

    # 保存
    pd.DataFrame(fc).to_csv(output_file, index=False, header=False)
    logging.info(f"✓ 示例FC矩阵已保存到: {output_file}")
    logging.info(f"  值范围: [{fc.min():.3f}, {fc.max():.3f}]")
    logging.info(f"  平均相关: {np.mean(fc[np.triu_indices(n_regions, k=1)]):.3f}")

    return fc


if __name__ == "__main__":
    # 示例用法
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*60)
    print("AAL-ALN Network 改进版示例")
    print("="*60 + "\n")

    # 创建示例FC矩阵
    fc_matrix = create_example_fc_matrix(n_regions=116)

    # 创建示例连接矩阵
    n_regions = 116
    np.random.seed(42)
    connectivity = np.random.rand(n_regions, n_regions) * 0.3
    connectivity = (connectivity + connectivity.T) / 2
    np.fill_diagonal(connectivity, 0)

    # 创建网络
    print("\n创建网络...")
    network = AALALNNetwork(
        connectivity_matrix=connectivity,
        n_regions=n_regions,
    )

    # 打印摘要
    network.print_summary()

    # 显示脑区信息
    region_info = network.get_region_info()
    print("前5个脑区信息:")
    print(region_info.head())

    print("\n✓ 示例运行完成！")
