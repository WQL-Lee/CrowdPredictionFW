# CrowdPredictFW - 机场人群预测框架

基于图神经网络的机场人群预测框架，支持多种时空图神经网络模型进行人群流量预测。

## 项目概述

CrowdPredictFW 是一个专门为机场人群预测设计的深度学习框架，集成了多种先进的图神经网络模型，包括：
- **TGCN** (Temporal Graph Convolutional Network)
- **AGCRN** (Adaptive Graph Convolutional Recurrent Network) 
- **A3TGCN** (Attention-based Temporal Graph Convolutional Network)
- **CrowdCNNGRU** (CNN-GRU混合模型)

## 项目结构

```
CrowdPredictFW/
├── config/                    # 配置文件目录
│   ├── train/                # 训练配置文件
│   ├── test/                 # 测试配置文件
│   └── vis/                  # 可视化配置文件
├── data/                     # 数据目录
├── data_loader/              # 数据加载器
│   ├── data_loaders.py       # 数据加载器主文件
│   ├── dataset.py           # 数据集基类
│   ├── TerminalDataset.py   # 航站楼数据集
│   └── TerminalFlightMixDataset.py  # 航站楼航班混合数据集
├── pred_model/              # 预测模型目录
│   ├── TGCN/                # TGCN模型实现
│   ├── AGCRN/               # AGCRN模型实现
│   ├── A3TGCN/              # A3TGCN模型实现
│   ├── structure/           # 传统CNN-GRU模型
│   ├── loss.py              # 损失函数
│   ├── metric.py            # 评估指标
├── trainer/                 # 训练器目录
│   ├── TGCNTrainer.py       # TGCN训练器
│   ├── AGCRNTrainer.py      # AGCRN训练器
│   └── A3TGCNTrainer.py     # A3TGCN训练器
├── utils/                   # 工具函数
├── logger/                  # 日志系统
├── train.py                 # 训练脚本
├── test.py                  # 测试脚本
├── vis.py                   # 可视化脚本
├── parse_config.py          # 配置解析器
└── requirements.txt         # 依赖包列表
```

## 环境配置

### 依赖安装

```bash
pip install -r requirements.txt
```

主要依赖包：
- `torch>=2.8.0` 
- `torch_geometric>=2.7.0` 
- `torch_geometric_temporal>=0.56.2` 
- `torchvision>=0.23.0` 
- `numpy>=2.3.4` 
- `pandas>=2.3.3`
- `matplotlib>=3.10.7` 
- `tqdm>=4.67.1` 
- `scipy>=1.16.2` 
- `h5py>=3.15.1` 
- `rich>=14.2.0`
- `yacs>=0.1.8` 

## 使用方法

### 1. 训练模型

#### 基本训练命令

```bash
python train.py -c config/train/TGCN.jsonc
```

#### 训练参数说明

- `-c, --config`: 指定配置文件路径
- `-r, --resume`: 指定要恢复的检查点路径
- `-d, --device`: 指定使用的GPU设备
- `-tr, --train`: 训练模式开关（默认True）

#### 自定义训练参数

```bash
python train.py -c config/train/TGCN.jsonc --lr 0.01 --bs 64
```

### 2. 测试模型

#### 基本测试命令

```bash
python test.py -c config/test/TGCN.jsonc
```

#### 测试参数说明

- `-c, --config`: 指定测试配置文件
- `-r, --resume`: 指定模型检查点路径
- `-d, --device`: 指定使用的GPU设备

### 3. 可视化结果

#### 基本可视化命令

```bash
python vis.py
```

或指定配置文件：

```bash
python vis.py -c config/vis/TGCN.jsonc
```

## 配置文件详解

### 训练配置文件 (config/train/)

#### 基本结构

```json
{
    "name": "model_name",                    // 模型名称
    "n_gpu": 1,                          // GPU数量
    "DEBUG": false,                      // 调试模式
    "result_info": {
        "is_saved": true,                // 是否保存结果
        "saved_dir": "result/model_name/train" // 结果保存目录
    },
    "loss": "loss_fn",                   // 损失函数
    "metrics": ["metric1", "metric2"],  // 评估指标
    "arch": {
        "type": "model_type",              // 模型类型，建议设置为模型名，即model_name
        "args": {
            "param": "value"              // 模型参数
        }                
    },
    "data_loader": {
        "type": "dataloader_type",         // 数据加载器类型，在data_loader下定义
        "args": {
            "param": "value"              // 数据加载参数
        }
    },
    "optimizer": {
        "type": "optimizer_type",          // 优化器类型
        "args": {
            "param": "value"              // 优化器参数
        }
    },
    "lr_scheduler": {
        "type": "scheduler_type",         // 学习率调度器类型
        "args": {
            "param": "value"              // 调度器参数
        }
    },
    "trainer": {
        "param": "value"                  // 训练参数
    }
}
```

#### 关键参数说明

**模型架构参数 (arch)**
- `type`: 模型类型，如 "TGCN", "AGCRN", "A3TGCN"
- `args`: 模型特定参数
  - `adj_path`: 邻接矩阵文件路径
  - `num_nodes`: 节点数量
  - `hidden_dim`: 隐藏层维度
  - `output_dim`: 输出维度

**数据加载器参数 (data_loader)**
- `type`: 数据加载器类型，如 "TerminalDataLoader"
- `args`: 数据加载参数
  - `data_dir`: 数据目录路径， [已处理的视频与航班信息人群数据集](https://drive.google.com/drive/folders/1mMpVHrdGh2-FFMMfXxiFIZbEtR9u-at3)
  - `batch_size`: 批次大小
  - `n_his`: 历史时间步数
  - `n_pred`: 预测时间步数
  - `is_continous`: 是否连续数据
  - `dates_dist`: 日期分布
  - `interval`: 时间间隔
  - `normalizer`: 归一化方法
  - `shuffle`: 是否打乱数据
  - `validation_split`: 验证集比例
  - `training`: 是否训练模式

**优化器参数 (optimizer)**
- `type`: 优化器类型，如 "Adam", "SGD"
- `args`: 优化器参数
  - `lr`: 学习率

**学习率调度器参数 (lr_scheduler)**
- `type`: 调度器类型，如 "StepLR", "CosineAnnealingLR"
- `args`: 调度器参数
  - `step_size`: 步长
  - `gamma`: 衰减因子

**训练器参数 (trainer)**
- `epochs`: 训练轮数
- `save_dir`: 模型保存目录
- `save_period`: 保存周期
- `verbosity`: 日志详细程度
- `early_stop`: 早停轮数
- `tensorboard`: 是否使用tensorboard

### 测试配置文件 (config/test/)

测试配置文件结构与训练配置类似，但包含以下特定参数：

```json
{
    "name": "model_name",               // 模型名称
    "DEBUG": false,                     // 调试模式
    "model_path": "path_to_model",      // 模型检查点路径
    "saved_dir": "path_to_saved",       // 测试结果保存目录
    "log_dir": "path_to_log",           // 测试日志目录
    "n_gpu": 1,                         // GPU数量
    "loss": "loss_function",            // 损失函数
    "metrics": ["metric1", "metric2"],   // 评估指标
    "data_loader": {
        "type": "dataloader_type",      // 数据加载器类型
        "args": {
            "data_dir": "data_path",     // 数据目录
            "batch_size": 1,             // 批次大小
            "n_his": 12,                 // 历史时间步数
            "n_pred": 6,                 // 预测时间步数
            "is_continous": true,        // 是否连续数据
            "dates_dist": [["start_date", "end_date"]], // 日期分布
            "interval": 5,               // 时间间隔
            "normalizer": "std",         // 归一化方法
            "shuffle": false,            // 是否打乱数据
            "validation_split": 0,       // 验证集比例
            "training": true             // 训练模式
        }
    },
    "arch": {
        "type": "model_type",           // 模型类型
        "args": {
            "adj_path": "adj_path",      // 邻接矩阵路径
            "num_nodes": 10,             // 节点数量
            "hidden_dim": 256,           // 隐藏层维度
            "output_dim": 6              // 输出维度
        }
    }
}
```

### 可视化配置文件 (config/vis/)

```json
{
    "name": "model_name",               // 模型名称
    "saved_dir": "vis_results_path",    // 可视化结果保存目录
    "loss_metrics": {
        "input_path": "loss_metrics_path", // 损失和指标文件路径
        "saved_sub_dir": "sub_dir_name",   // 子目录名称
        "loss": {
            "dynamic": false,              // 是否动态显示
            "figsize": [8,4],              // 图像尺寸
            "color": "black",              // 线条颜色
            "linewidth": 2,                // 线条宽度
            "dpi": 100                     // 图像分辨率
        },
        "metrics": {
            "strategy": "all",             // 显示策略
            "color": "red",                // 线条颜色
            "linewidth": 2,                // 线条宽度
            "dpi": 100,                    // 图像分辨率
            "selected_keys": ["RMSE"],     // 选中的指标
            "s_figsize": [8, 4],          // 单图尺寸
            "keys": ["RMSE","RMSE"],      // 指标键名
            "ncols": 2,                    // 列数
            "nrows": 1,                    // 行数
            "power_limits": [1, 4],        // 幂次限制
            "a_figsize": [8, 4]           // 所有图尺寸
        }
    },
    "tgt_pred": {
        "input_path": "pred_results_path", // 预测结果文件路径
        "saved_sub_dir": "tgt_pred_dir",   // 目标预测可视化子目录
        "title": ["area1","area2","area3"], // 区域标题列表
        "figsize": [20,50],               // 图像尺寸
        "dpi": 100                        // 图像分辨率
    }
}
```

## 如何新增预测模型

### 1. 创建模型文件

在 `pred_model/` 目录下创建新的模型目录，例如 `MyModel/`：

```
pred_model/MyModel/
├── __init__.py
├── MyModel.py          # 模型实现
└── utils.py            # 工具函数（可选）
```

### 2. 实现模型类

在 `MyModel.py` 中实现模型类：

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, adj_path, num_nodes, hidden_dim, output_dim, **kwargs):
        super(MyModel, self).__init__()
        self.name = "MyModel"  # 设置模型名称
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 加载邻接矩阵
        self.adj = self.load_adj(adj_path)
        
        # 定义模型层
        self.layer1 = nn.Linear(num_nodes, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def load_adj(self, adj_path):
        # 实现邻接矩阵加载逻辑
        import pandas as pd
        adj = pd.read_csv(adj_path, header=None).values
        return torch.FloatTensor(adj)
        
    def forward(self, x):
        # 实现前向传播逻辑
        # x shape: (batch_size, seq_len, num_nodes)
        batch_size, seq_len, num_nodes = x.shape
        
        # 处理时序数据
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, num_nodes)
            x_t = self.layer1(x_t)
            x_t = torch.relu(x_t)
            x_t = self.layer2(x_t)
            outputs.append(x_t)
        
        # 堆叠输出 (batch_size, seq_len, output_dim)
        output = torch.stack(outputs, dim=1)
        return output
```

### 3. 创建训练器

在 `trainer/` 目录下创建 `MyModelTrainer.py`：

```python
import torch
import numpy as np
from base import BaseTrainer
from utils import inf_loop, MetricTracker

class MyModelTrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            self.len_epoch = len(self.data_loader)
        else:
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        
        self.train_per_epoch = len(data_loader)
        if valid_data_loader is not None:
            self.valid_per_epoch = len(valid_data_loader)
        
        self.model_name = model.name
        self.scaler = self.data_loader.dataset.scaler

        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
    
    def _train_epoch(self, epoch):
        """训练一个epoch的逻辑"""
        total_loss = 0.0
        self.model.train()
        
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch, self._progress(batch_idx), loss.item()))
        
        return {'loss': total_loss / len(self.data_loader)}
    
    def _valid_epoch(self, epoch):
        """验证一个epoch的逻辑"""
        self.model.eval()
        self.valid_metrics.reset()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
        
        return self.valid_metrics.result()
    
    @staticmethod
    def test(model, data_loader, device, scaler):
        """测试模型"""
        model.eval()
        y_pred_list = []
        y_true_list = []
        y_time_list = []
        
        with torch.no_grad():
            for batch_idx, (data, target, time_info) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                # 反归一化
                if scaler is not None:
                    output = scaler.inverse_transform(output.cpu().numpy())
                    target = scaler.inverse_transform(target.cpu().numpy())
                else:
                    output = output.cpu().numpy()
                    target = target.cpu().numpy()
                
                y_pred_list.append(output)
                y_true_list.append(target)
                y_time_list.append(time_info)
        
        return np.concatenate(y_pred_list), np.concatenate(y_true_list), np.concatenate(y_time_list)
```

### 4. 创建配置文件

在 `config/train/` 目录下创建 `MyModel.jsonc`：

```json
{
    "name": "MyModel",                    // 模型名称
    "n_gpu": 1,                          // GPU数量
    "DEBUG": false,                      // 调试模式
    "result_info": {
        "is_saved": true,                // 是否保存结果
        "saved_dir": "result/MyModel/train"  // 结果保存目录
    },
    "loss": "mse_loss",                  // 损失函数
    "metrics": ["RMSE", "MAE"],           // 评估指标
    "arch": {
        "type": "MyModel",               // 模型类型
        "args": {
            "adj_path": "data/temp/adj.csv",  // 邻接矩阵路径
            "num_nodes": 10,                  // 节点数量
            "hidden_dim": 256,                // 隐藏层维度
            "output_dim": 6                   // 输出维度
        }
    },
    "data_loader": {
        "type": "TerminalDataLoader",    // 数据加载器类型
        "args": {
            "data_dir": "data/temp",           // 数据目录
            "batch_size": 32,                  // 批次大小
            "n_his": 12,                       // 历史时间步数
            "n_pred": 6,                       // 预测时间步数
            "is_continous": true,              // 是否连续数据
            "dates_dist": [["20240609", "20240618"]],  // 日期分布
            "interval": 5,                     // 时间间隔
            "normalizer": "std",               // 归一化方法
            "shuffle": true,                   // 是否打乱数据
            "validation_split": 0.2,           // 验证集比例
            "training": true                   // 训练模式
        }
    },
    "optimizer": {
        "type": "Adam",                    // 优化器类型
        "args": {
            "lr": 0.001                     // 学习率
        }
    },
    "lr_scheduler": {
        "type": "StepLR",                 // 学习率调度器类型
        "args": {
            "step_size": 1000,            // 步长
            "gamma": 0.1                  // 衰减因子
        }
    },
    "trainer": {
        "epochs": 100,                    // 训练轮数
        "save_dir": "LSaved/MyModel",     // 模型保存目录
        "save_period": 10,                // 保存周期
        "verbosity": 2,                  // 日志详细程度
        "early_stop": 20,                // 早停轮数
        "tensorboard": true              // 是否使用tensorboard
    }
}
```

### 5. 更新训练脚本

在 `train.py` 中添加新模型的导入：

```python
# 取消注释或添加新模型
import pred_model.MyModel.MyModel as module_arch
from trainer import MyModelTrainer as Trainer
```

### 6. 更新测试脚本

在 `test.py` 中添加新模型的测试逻辑：

```python
elif self.model_name == 'MyModel':
    y_pred, y_true, y_time = MyModelTrainer.test(self.model, self.data_loader, self.device, self.scaler)
```

## 目录作用说明

### 核心目录

- **`config/`**: 存放所有配置文件，包括训练、测试、可视化配置
  - `train/`: 训练配置文件，定义模型架构、数据加载、优化器等参数
  - `test/`: 测试配置文件，定义测试数据和模型路径
  - `vis/`: 可视化配置文件，定义可视化参数和输出路径

- **`data/`**: 存放原始数据和预处理后的数据

- **`data_loader/`**: 数据加载器实现，负责数据预处理和批次生成
  - `data_loaders.py`: 数据加载器主文件，定义数据加载器工厂
  - `dataset.py`: 数据集基类，定义通用数据集接口
  - `TerminalDataset.py`: 航站楼数据集，处理航站楼人群数据
  - `TerminalFlightMixDataset.py`: 航站楼航班混合数据集

- **`pred_model/`**: 所有预测模型的实现
  - `TGCN/`: TGCN模型实现
  - `AGCRN/`: AGCRN模型实现
  - `A3TGCN/`: A3TGCN模型实现
  - `structure/`: 传统CNN-GRU模型
  - `loss.py`: 损失函数实现
  - `metric.py`: 评估指标实现

- **`trainer/`**: 训练器实现，包含不同模型的训练逻辑
  - `TGCNTrainer.py`: TGCN训练器
  - `AGCRNTrainer.py`: AGCRN训练器
  - `A3TGCNTrainer.py`: A3TGCN训练器

- **`utils/`**: 工具函数和辅助类
- **`logger/`**: 日志系统实现



### 可视化目录

- **`vis/`**: 可视化工具和脚本
- **`vis.py`**: 主要可视化脚本

## 数据格式要求

### 输入数据格式

1. **邻接矩阵文件** (`adj.csv`): CSV格式的邻接矩阵，定义节点间的连接关系
2. **时间序列数据**: 按时间顺序组织的人群流量数据
3. **航班数据**: 航班信息数据（可选）

### 数据目录结构

```
data/
├── temp/                    # 临时数据目录
│   ├── adj.csv             # 邻接矩阵
│   ├── crowd_data.csv      # 人群流量数据
│   └── flight_data.csv     # 航班数据（可选）
```

### 数据格式说明

- **邻接矩阵**: N×N的矩阵，N为节点数量，值表示节点间的连接强度
- **人群数据**: 时间序列数据，包含多个区域的人群流量
- **时间格式**: 支持多种时间格式，如"20240609"、"2024-06-09"等

## 评估指标

框架支持多种评估指标：

- **RMSE**: 均方根误差 (Root Mean Square Error)
- **MAE**: 平均绝对误差 (Mean Absolute Error)
- **MAPE**: 平均绝对百分比误差 (Mean Absolute Percentage Error)
- **Accuracy**: 准确率
- **R2**: 决定系数 (R-squared)
- **Explained_Variance**: 解释方差

## 可视化功能

### 损失和指标可视化

- 训练损失曲线
- 验证指标变化
- 多指标对比图
- 支持动态显示和静态保存

### 预测结果可视化

- 真实值vs预测值对比
- 多区域预测结果
- 时间序列预测图
- 支持自定义图像尺寸和分辨率

## 常见问题

### 1. 内存不足

- 减小 `batch_size`
- 减少 `n_his` 和 `n_pred` 参数
- 使用梯度累积
- 调整 `hidden_dim` 参数

### 2. 训练速度慢

- 使用GPU训练
- 调整数据加载器参数
- 使用混合精度训练
- 减少模型复杂度

### 3. 模型不收敛

- 调整学习率
- 检查数据预处理
- 调整模型架构参数
- 使用不同的优化器

### 4. 数据加载问题

- 检查数据路径是否正确
- 确认数据格式是否符合要求
- 检查日期格式是否正确
- 验证邻接矩阵维度

## 高级用法

### 1. 自定义损失函数

在 `pred_model/loss.py` 中添加新的损失函数：

```python
def custom_loss(output, target):
    # 实现自定义损失函数
    return loss_value
```

### 2. 自定义评估指标

在 `pred_model/metric.py` 中添加新的评估指标：

```python
def custom_metric(output, target):
    # 实现自定义评估指标
    return metric_value
```

**注意**: 使用前请确保已正确安装所有依赖包，并根据实际数据格式调整配置文件参数。建议先在小型数据集上测试模型，确认配置正确后再进行大规模训练。