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
│   └── n_metric.py          # 归一化评估指标
├── trainer/                 # 训练器目录
│   ├── TGCNTrainer.py       # TGCN训练器
│   ├── AGCRNTrainer.py      # AGCRN训练器
│   └── A3TGCNTrainer.py     # A3TGCN训练器
├── utils/                   # 工具函数
├── logger/                  # 日志系统
├── vis/                     # 可视化工具
├── result/                  # 结果保存目录
├── LSaved/                  # 模型保存目录
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
- `torch>=1.1` - PyTorch深度学习框架
- `torchvision` - 计算机视觉工具
- `numpy` - 数值计算
- `tqdm` - 进度条
- `tensorboard>=1.14` - 训练可视化
- `torch_geometric` - 图神经网络库
- `matplotlib` - 绘图库
- `pandas` - 数据处理

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
    "name": "模型名称",
    "n_gpu": 1,
    "DEBUG": false,
    "result_info": {
        "is_saved": true,
        "saved_dir": "result/模型名/train"
    },
    "loss": "损失函数名称",
    "metrics": ["评估指标列表"],
  "arch": {
        "type": "模型类型",
    "args": {
            "模型参数": "参数值"
    }                
  },
  "data_loader": {
        "type": "数据加载器类型",
        "args": {
            "数据加载参数": "参数值"
    }
  },
  "optimizer": {
        "type": "优化器类型",
        "args": {
            "优化器参数": "参数值"
        }
    },
  "lr_scheduler": {
        "type": "学习率调度器类型",
        "args": {
            "调度器参数": "参数值"
    }
  },
  "trainer": {
        "训练参数": "参数值"
  }
}
```

#### 关键参数说明

**模型架构参数 (arch)**
- `type`: 模型类型，如 "TGCN", "AGCRN", "A3TGCN"
- `args`: 模型特定参数
  - `adj_path`: 邻接矩阵文件路径 (adjacency matrix file path)
  - `num_nodes`: 节点数量 (number of nodes)
  - `hidden_dim`: 隐藏层维度 (hidden dimension)
  - `output_dim`: 输出维度 (output dimension)

**数据加载器参数 (data_loader)**
- `type`: 数据加载器类型，如 "TerminalDataLoader"
- `args`: 数据加载参数
  - `data_dir`: 数据目录路径 (data directory path)
  - `batch_size`: 批次大小 (batch size)
  - `n_his`: 历史时间步数 (number of historical time steps)
  - `n_pred`: 预测时间步数 (number of prediction time steps)
  - `is_continous`: 是否连续数据 (whether data is continuous)
  - `dates_dist`: 日期分布 (date distribution)
  - `interval`: 时间间隔 (time interval)
  - `normalizer`: 归一化方法 (normalization method)
  - `shuffle`: 是否打乱数据 (whether to shuffle data)
  - `validation_split`: 验证集比例 (validation split ratio)
  - `training`: 是否训练模式 (training mode flag)

**优化器参数 (optimizer)**
- `type`: 优化器类型，如 "Adam", "SGD"
- `args`: 优化器参数
  - `lr`: 学习率 (learning rate)

**学习率调度器参数 (lr_scheduler)**
- `type`: 调度器类型，如 "StepLR", "CosineAnnealingLR"
- `args`: 调度器参数
  - `step_size`: 步长 (step size)
  - `gamma`: 衰减因子 (decay factor)

**训练器参数 (trainer)**
- `epochs`: 训练轮数 (number of training epochs)
- `save_dir`: 模型保存目录 (model save directory)
- `save_period`: 保存周期 (save period)
- `verbosity`: 日志详细程度 (logging verbosity level)
- `early_stop`: 早停轮数 (early stopping patience)
- `tensorboard`: 是否使用tensorboard (whether to use tensorboard)

### 测试配置文件 (config/test/)

测试配置文件结构与训练配置类似，但包含以下特定参数：

```json
{
    "model_path": "模型检查点路径 (model checkpoint path)",
    "saved_dir": "测试结果保存目录 (test results save directory)"
}
```

### 可视化配置文件 (config/vis/)

```json
{
    "name": "模型名称 (model name)",
    "saved_dir": "可视化结果保存目录 (visualization results save directory)",
    "loss_metrics": {
        "input_path": "损失和指标文件路径 (loss and metrics file path)",
        "saved_sub_dir": "子目录名称 (subdirectory name)",
        "loss": {
            "dynamic": false,  // 是否动态显示 (whether to display dynamically)
            "figsize": [8,4],  // 图像尺寸 (figure size)
            "color": "black",  // 线条颜色 (line color)
            "linewidth": 2,    // 线条宽度 (line width)
            "dpi": 100         // 图像分辨率 (image resolution)
        },
        "metrics": {
            "strategy": "all",  // 显示策略 (display strategy)
            "color": "red",     // 线条颜色 (line color)
            "linewidth": 2,     // 线条宽度 (line width)
            "dpi": 100,         // 图像分辨率 (image resolution)
            "selected_keys": ["RMSE"],  // 选中的指标 (selected metrics)
            "s_figsize": [8, 4],        // 单图尺寸 (single figure size)
            "keys": ["RMSE","RMSE"],    // 指标键名 (metric keys)
            "ncols": 2,                 // 列数 (number of columns)
            "nrows": 1,                 // 行数 (number of rows)
            "power_limits": [1, 4],     // 幂次限制 (power limits)
            "a_figsize": [8, 4]        // 所有图尺寸 (all figures size)
        }
    },
    "tgt_pred": {
        "input_path": "预测结果文件路径 (prediction results file path)",
        "saved_sub_dir": "目标预测可视化子目录 (target prediction visualization subdirectory)",
        "title": ["区域标题列表 (area title list)"],
        "figsize": [20,50],  // 图像尺寸 (figure size)
        "dpi": 100           // 图像分辨率 (image resolution)
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
        pass
        
    def forward(self, x):
        # 实现前向传播逻辑
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x
```

### 3. 创建训练器

在 `trainer/` 目录下创建 `MyModelTrainer.py`：

  ```python
import torch
from trainer.base_trainer import BaseTrainer

class MyModelTrainer(BaseTrainer):
    def __init__(self, model, criterion, metrics, optimizer, config, device, data_loader, valid_data_loader, lr_scheduler):
        super().__init__(model, criterion, metrics, optimizer, config, device, data_loader, valid_data_loader, lr_scheduler)
    
    def _train_epoch(self, epoch):
        # 实现训练逻辑
        pass
    
    def _valid_epoch(self, epoch):
        # 实现验证逻辑
        pass
    
    @staticmethod
    def test(model, data_loader, device, scaler):
        # 实现测试逻辑
      pass
  ```

### 4. 创建配置文件

在 `config/train/` 目录下创建 `MyModel.jsonc`：

  ```json
{
    "name": "MyModel",  // 模型名称 (model name)
    "n_gpu": 1,         // GPU数量 (number of GPUs)
    "DEBUG": false,     // 调试模式 (debug mode)
    "result_info": {
        "is_saved": true,                    // 是否保存结果 (whether to save results)
        "saved_dir": "result/MyModel/train"  // 结果保存目录 (results save directory)
    },
    "loss": "mse_loss",  // 损失函数 (loss function)
    "metrics": ["RMSE", "MAE"],  // 评估指标 (evaluation metrics)
    "arch": {
        "type": "MyModel",  // 模型类型 (model type)
        "args": {
            "adj_path": "data/temp/adj.csv",  // 邻接矩阵路径 (adjacency matrix path)
            "num_nodes": 10,                  // 节点数量 (number of nodes)
            "hidden_dim": 256,                // 隐藏层维度 (hidden dimension)
            "output_dim": 6                   // 输出维度 (output dimension)
        }
    },
    "data_loader": {
        "type": "TerminalDataLoader",  // 数据加载器类型 (data loader type)
        "args": {
            "data_dir": "data/temp",           // 数据目录 (data directory)
            "batch_size": 32,                  // 批次大小 (batch size)
            "n_his": 12,                       // 历史时间步数 (historical time steps)
            "n_pred": 6,                       // 预测时间步数 (prediction time steps)
            "is_continous": true,              // 是否连续数据 (continuous data flag)
            "dates_dist": [["20240609", "20240618"]],  // 日期分布 (date distribution)
            "interval": 5,                     // 时间间隔 (time interval)
            "normalizer": "std",               // 归一化方法 (normalization method)
            "shuffle": true,                   // 是否打乱数据 (shuffle data)
            "validation_split": 0.2,           // 验证集比例 (validation split ratio)
            "training": true                    // 训练模式 (training mode)
        }
    },
    "optimizer": {
        "type": "Adam",  // 优化器类型 (optimizer type)
        "args": {
            "lr": 0.001  // 学习率 (learning rate)
        }
    },
    "lr_scheduler": {
        "type": "StepLR",  // 学习率调度器类型 (learning rate scheduler type)
        "args": {
            "step_size": 1000,  // 步长 (step size)
            "gamma": 0.1         // 衰减因子 (decay factor)
        }
    },
    "trainer": {
        "epochs": 100,                    // 训练轮数 (number of epochs)
        "save_dir": "LSaved/MyModel",    // 模型保存目录 (model save directory)
        "save_period": 10,                // 保存周期 (save period)
        "verbosity": 2,                  // 日志详细程度 (logging verbosity)
        "early_stop": 20,                // 早停轮数 (early stopping patience)
        "tensorboard": true              // 是否使用tensorboard (use tensorboard)
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
- **`data/`**: 存放原始数据和预处理后的数据
- **`data_loader/`**: 数据加载器实现，负责数据预处理和批次生成
- **`pred_model/`**: 所有预测模型的实现
- **`trainer/`**: 训练器实现，包含不同模型的训练逻辑
- **`utils/`**: 工具函数和辅助类
- **`logger/`**: 日志系统实现

### 结果目录

- **`result/`**: 存放训练结果、测试结果和可视化结果
- **`LSaved/`**: 存放训练好的模型检查点
- **`TestLog/`**: 存放测试日志

### 可视化目录

- **`vis/`**: 可视化工具和脚本
- **`vis.py`**: 主要可视化脚本

## 数据格式要求

### 输入数据格式

1. **邻接矩阵文件** (`adj.csv`): CSV格式的邻接矩阵
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

## 评估指标

框架支持多种评估指标：

- **RMSE**: 均方根误差
- **MAE**: 平均绝对误差
- **Accuracy**: 准确率
- **R2**: 决定系数
- **Explained_Variance**: 解释方差

## 可视化功能

### 损失和指标可视化

- 训练损失曲线
- 验证指标变化
- 多指标对比图

### 预测结果可视化

- 真实值vs预测值对比
- 多区域预测结果
- 时间序列预测图

## 常见问题

### 1. 内存不足

- 减小 `batch_size`
- 减少 `n_his` 和 `n_pred` 参数
- 使用梯度累积

### 2. 训练速度慢

- 使用GPU训练
- 调整数据加载器参数
- 使用混合精度训练

### 3. 模型不收敛

- 调整学习率
- 检查数据预处理
- 调整模型架构参数

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 创建 Issue
- 发送邮件至项目维护者

---

**注意**: 使用前请确保已正确安装所有依赖包，并根据实际数据格式调整配置文件参数。