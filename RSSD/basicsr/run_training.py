#!/usr/bin/env python3
"""
MambaIR 训练启动脚本
支持图像修复和超分联合训练
"""

import os
import sys
import argparse
from pathlib import Path


def setup_environment():
    """设置运行环境"""
    # 获取 basicsr 的父目录（即项目根目录）
    current_file = Path(__file__).resolve()
    basicsr_dir = current_file.parent
    project_root = basicsr_dir.parent
    
    # 将项目根目录添加到 Python 路径
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 设置 CUDA 相关环境变量
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    print(f"项目根目录: {project_root}")
    print(f"basicsr 目录: {basicsr_dir}")
    print(f"Python 路径已更新: {sys.path[0]}")
    
    return project_root


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MambaIR 训练脚本')
    parser.add_argument('--config', type=str,
                        default='options/train/inpating/combined_config.yml',
                        help='配置文件路径')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='GPU设备ID，用逗号分隔 (例: 0,1)')
    parser.add_argument('--mode', type=int, default=1, choices=[1, 2],
                        help='运行模式: 1=训练, 2=测试')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    return parser.parse_args()


def main():
    """主函数"""
    # 设置环境
    project_root = setup_environment()
    args = parse_arguments()

    # 现在可以安全导入 basicsr 模块
    try:
        from basicsr.train import main as train_main
    except ImportError as e:
        print(f"导入错误: {e}")
        print(f"当前 Python 路径: {sys.path}")
        print(f"请确保在项目根目录运行此脚本")
        sys.exit(1)

    # 修改 sys.argv 以传递参数
    sys.argv = [
        'train.py',
        '--config', args.config,
        '--gpu_ids', args.gpu_ids,
        '--mode', str(args.mode)
    ]

    if args.resume:
        sys.argv.extend(['--resume', args.resume])
    if args.debug:
        sys.argv.append('--debug')

    print("=" * 80)
    print("开始训练 MambaIR 模型...")
    print(f"配置文件: {args.config}")
    print(f"GPU 设备: {args.gpu_ids}")
    print(f"运行模式: {'训练' if args.mode == 1 else '测试'}")
    print("=" * 80)

    try:
        train_main()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
