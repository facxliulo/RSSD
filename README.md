## 📁 项目结构

```
RSSD\main\
├── basicsr/                 # 主要代码目录
│   ├── __init__.py             # 包初始化文件（重要！）
│   ├── train.py                # 训练主程序（已修复）
│   ├── archs/                  # 网络架构
│   ├── models/                 # 模型定义
│   ├── data/                   # 数据加载
│   ├── utils/                  # 工具函数
│   ├── options/                # 配置系统
│   └── trains/                 # 训练器
├── options/                    # 配置文件目录
│   └── train/
│       └── inpating/
│           └── combined_config.yml
├── start_training.py           # ✨ 新的启动脚本（推荐）
├── run_train.py               # 备用启动脚本
└── README_启动指南.md          # 本文件
```

---

## 🎯 快速开始

### 方法 1: 使用新启动脚本（推荐⭐）

```bash
# 1. 进入项目根目录
cd "RSSD\main"

# 2. 运行训练（会自动处理路径问题）
python start_training.py --config options/train/inpating/combined_config.yml
```

### 方法 2: 使用备用脚本

```bash
cd "RSSD\main"
python run_train.py --config options/train/inpating/combined_config.yml
```

### 方法 3: 直接运行（仅在路径正确时）

```bash
cd "RSSD\main"
python basicsr/run_training.py --config options/train/inpating/combined_config.yml
```

---

## 🔧 命令行参数

```bash
python start_training.py \
    --config options/train/inpating/combined_config.yml \  # 配置文件路径
    --gpu_ids 0 \                                         # GPU设备ID  
    --mode 1                                              # 1=训练, 2=测试
```
## We will upload the revised version of the paper shortly.
```bash
cd "RSSD\main"
python start_training.py --config options/train/inpating/combined_config.yml
```

现在问题应该完全解决了！🎉
