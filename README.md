[README_启动指南.md](https://github.com/user-attachments/files/28001140/README_.md)
# 🚀 MambaIR 项目启动指南

## ✅ 问题已解决！

之前遇到的 `ModuleNotFoundError: No module named 'basicsr'` 错误已经通过以下方式解决：

### 解决方案概述

1. **路径设置修复** - 确保项目根目录正确添加到 Python 搜索路径
2. **导入语句优化** - 添加详细的错误诊断和调试信息  
3. **新启动脚本** - 创建了更简单、更健壮的启动脚本

---

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

---

## 🐛 故障排查

### 问题 1: 仍然提示 `No module named 'basicsr'`

**解决方法:**

```bash
# 检查当前目录
pwd  # 应该显示: RSSD\main

# 检查 basicsr 是否存在
ls -la basicsr/

# 检查 __init__.py 是否存在
ls -la basicsr/__init__.py
```

**如果还是不行，使用绝对导入:**

```python
# 临时解决方案：在运行前手动设置路径
import sys
sys.path.insert(0, r'RSSD\main')
```

### 问题 2: 配置文件找不到

```bash
# 检查配置文件是否存在
ls -la options/train/inpating/combined_config.yml

# 如果不存在，使用默认配置创建
python create_default_config.py
```

### 问题 3: CUDA 相关错误

```bash
# 检查 CUDA 是否可用
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 如果 CUDA 不可用，在配置文件中设置 CPU 模式
# 或使用环境变量
export CUDA_VISIBLE_DEVICES=""
```

### 问题 4: 依赖包缺失

```bash
# 检查依赖
pip list | grep -E "torch|mamba|basicsr"

# 安装缺失的包
pip install -r requirements.txt
```

---

## 📝 详细启动流程说明

### start_training.py 的工作原理

```python
# 1. 设置项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 2. 添加到 Python 路径（这是关键！）
sys.path.insert(0, PROJECT_ROOT)

# 3. 现在可以正常导入了
from basicsr.train import main

# 4. 启动训练
main()
```

### 为什么之前会失败？

1. **错误的工作目录**: 从 `basicsr/` 目录内运行脚本
2. **Python 路径未设置**: `basicsr` 的父目录不在 `sys.path` 中
3. **相对导入问题**: 使用了不正确的相对导入路径

### 修复后的改进

1. ✅ **自动路径检测**: 脚本自动找到项目根目录
2. ✅ **路径验证**: 启动前检查必要的目录和文件
3. ✅ **详细错误信息**: 如果出错会显示具体原因和解决方法
4. ✅ **向后兼容**: 支持多种启动方式

---

## 🎓 最佳实践

### ✅ 推荐做法

```bash
# 1. 总是从项目根目录运行
cd "RSSD\main"
python start_training.py --config <配置文件>

# 2. 使用绝对路径指定配置文件
python start_training.py --config "$(pwd)/options/train/inpating/combined_config.yml"

# 3. 使用虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
python start_training.py --config <配置文件>
```

### ❌ 避免的做法

```bash
# ❌ 不要在子目录中运行
cd basicsr
python train.py  # 这样会出错！

# ❌ 不要使用相对路径从其他目录运行
cd /some/other/path
python "RSSD\main\basicsr\train.py"  # 路径会不对！
```

---

## 📊 运行示例输出

成功启动时应该看到:

```
====================================================================================================
🚀 MambaIR 训练启动器
====================================================================================================
📁 项目根目录: RSSD\main
🐍 Python版本: 3.9.0
📦 Python路径已更新: RSSD\main
====================================================================================================

📋 检查项目结构...
  ✅ basicsr/ 存在
  ✅ options/ 存在

📦 导入训练模块...
  ✅ basicsr 模块路径: ['RSSD\\main\\main\\basicsr']
  ✅ 所有模块导入成功!

====================================================================================================
🎯 开始训练...
====================================================================================================

[train.py] 当前目录: RSSD\main\basicsr
[train.py] 父目录: RSSD\main
[train.py] Python 路径: ['RSSD\\main', ...]
[train.py] ✓ 成功导入 UnifiedConfig
[train.py] ✓ 成功导入 InpaintingTrainer
[train.py] ✓ 成功导入 InpaintingModel
[train.py] ✓ 成功导入 MambaIRv2
[train.py] ✓ 成功导入 CombinedDataset
[train.py] ✓ 成功导入 build_dataloader
[train.py] ✓ 成功导入 utils 模块
[train.py] 所有模块导入成功!
================================================================================
```

---

## 🆘 仍然遇到问题？

如果上述方法都无法解决问题，请提供以下信息：

1. **当前工作目录**:
   ```bash
   pwd  # 或 cd (Windows)
   ```

2. **Python 路径**:
   ```python
   python -c "import sys; print('\n'.join(sys.path))"
   ```

3. **目录结构**:
   ```bash
   ls -la basicsr/
   ```

4. **完整错误信息**:
   ```bash
   python start_training.py --config <配置文件> 2>&1 | tee error.log
   ```

---

## ✨ 总结

**核心要点:**
1. 始终从项目根目录 `RSSD\main` 运行脚本
2. 使用新创建的 `start_training.py` 启动脚本
3. 确保 `basicsr/__init__.py` 文件存在
4. 如果修改了代码，确保路径设置正确

**快速命令:**
```bash
cd "RSSD\main"
python start_training.py --config options/train/inpating/combined_config.yml
```

现在问题应该完全解决了！🎉
