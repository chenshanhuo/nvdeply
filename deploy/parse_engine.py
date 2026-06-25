#!/usr/bin/env python3
"""
脚本名称：parse_engine.py
功能：执行 trtexec 命令，支持动态形状的TensorRT引擎分析
参数要求：
  --loadEngine: 必选，路径到 .engine 文件
  --exportLayerInfo: 必选，输出JSON图结构文件
  --shapes: 可选，指定输入形状，格式为"input_name:BxHxW"（用于动态引擎）
  --minShapes, --optShapes, --maxShapes: 可选，动态形状范围
  --iterations: 可选，迭代次数
  --warmUp: 可选，预热次数
用法示例：
  python parse_engine.py --loadEngine model.engine --exportLayerInfo model_graph.json
  python parse_engine.py --loadEngine dynamic_model.engine --exportLayerInfo graph.json --shapes input:1x3x520x520
  python parse_engine.py --loadEngine dynamic_model.engine --exportLayerInfo graph.json --minShapes input:1x3x320x320 --optShapes input:1x3x520x520 --maxShapes input:1x3x1024x1024
"""

import argparse
import subprocess
import os
import sys
import json
from pathlib import Path
import re

def parse_shape_string(shape_str):
    """
    解析形状字符串，格式为"input_name:BxHxW" 或 "BxHxW"
    返回字典：{'name': 'input_name', 'shape': [B, H, W]}
    """
    if ':' in shape_str:
        name, shape_part = shape_str.split(':', 1)
    else:
        name = "input"  # 默认输入名
        shape_part = shape_str
    
    # 解析形状部分
    try:
        shape = [int(dim) for dim in shape_part.split('x')]
    except ValueError as e:
        raise ValueError(f"无效的形状格式: {shape_str}。应为 'name:BxHxW' 或 'BxHxW' 格式")
    
    return {'name': name, 'shape': shape}

def build_trtexec_command(args):
    """构建 trtexec 命令参数列表"""
    cmd = ["trtexec"]
    
    # 必需参数
    cmd.extend(["--loadEngine", args.loadEngine])
    cmd.extend(["--exportLayerInfo", args.exportLayerInfo])
    
    # 性能测试参数
    if args.iterations:
        cmd.extend(["--iterations", str(args.iterations)])
    else:
        cmd.extend(["--iterations", "100"])  # 默认100次迭代
    
    if args.warmUp:
        cmd.extend(["--warmUp", str(args.warmUp)])
    else:
        cmd.extend(["--warmUp", "10"])  # 默认10次预热
    
    # 形状参数处理
    if args.shapes:
        # 单个形状模式
        shape_info = parse_shape_string(args.shapes)
        cmd.extend(["--shapes", f"{shape_info['name']}:{'x'.join(map(str, shape_info['shape']))}"])
    
    elif args.minShapes or args.optShapes or args.maxShapes:
        # 动态形状范围模式
        if not (args.minShapes and args.optShapes and args.maxShapes):
            print("⚠️  警告: 使用动态形状范围时，需要同时指定 --minShapes, --optShapes, --maxShapes")
            print("   将尝试使用现有参数继续执行...")
        
        for shape_type, shape_str in [("minShapes", args.minShapes), 
                                      ("optShapes", args.optShapes), 
                                      ("maxShapes", args.maxShapes)]:
            if shape_str:
                shape_info = parse_shape_string(shape_str)
                cmd.extend([f"--{shape_type}", f"{shape_info['name']}:{'x'.join(map(str, shape_info['shape']))}"])
    
    else:
        # 尝试自动检测动态形状
        print("ℹ️  未指定形状参数，将使用默认形状 1x3x520x520")
        print("   如果模型是动态的，建议使用 --shapes 参数指定形状")
        cmd.extend(["--shapes", "input:1x3x520x520"])
    
    # 添加其他优化参数
    cmd.extend(["--useCudaGraph"])  # 启用CUDA图加速
    cmd.extend(["--noDataTransfers"])  # 不进行数据传输，减少开销
    
    return cmd

def analyze_engine_metadata(engine_path):
    """分析引擎文件的元数据"""
    print(f"\n🔍 分析引擎文件: {engine_path}")
    
    if not os.path.exists(engine_path):
        print(f"❌ 引擎文件不存在: {engine_path}")
        return None
    
    file_size = os.path.getsize(engine_path) / 1024 / 1024
    print(f"   文件大小: {file_size:.2f} MB")
    
    # 检查文件头，判断是否为TensorRT引擎文件
    with open(engine_path, 'rb') as f:
        header = f.read(4)
        if header == b'\x1bNTrt':
            print("   文件类型: TensorRT 引擎 (版本 >= 8.6)")
        else:
            print("   文件类型: TensorRT 引擎 (旧版本)")
    
    return file_size

def parse_trtexec_output(output):
    """解析 trtexec 输出，提取关键性能指标"""
    metrics = {
        'latency': None,
        'throughput': None,
        'memory_usage': None
    }
    
    lines = output.split('\n')
    for i, line in enumerate(lines):
        # 提取延迟信息
        if 'Latency:' in line:
            # 查找类似 "Latency: min = 2.43457 ms, max = 3.12345 ms, mean = 2.56789 ms"
            latency_match = re.search(r'mean = ([\d.]+) ms', line)
            if latency_match:
                metrics['latency'] = float(latency_match.group(1))
        
        # 提取吞吐量信息
        elif 'Throughput:' in line:
            # 查找类似 "Throughput: 389.123 qps"
            throughput_match = re.search(r'Throughput: ([\d.]+) qps', line)
            if throughput_match:
                metrics['throughput'] = float(throughput_match.group(1))
        
        # 提取内存使用信息
        elif 'GPU Compute' in line and 'Peak Memory Usage' in lines[i+1]:
            mem_match = re.search(r'GPU: ([\d.]+) MiB', lines[i+1])
            if mem_match:
                metrics['memory_usage'] = float(mem_match.group(1))
    
    return metrics

def main():
    parser = argparse.ArgumentParser(
        description='TensorRT引擎分析工具 - 支持动态形状引擎',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法 (静态或动态引擎)
  %(prog)s --loadEngine model.engine --exportLayerInfo model_graph.json
  
  # 动态引擎指定固定形状
  %(prog)s --loadEngine dynamic_model.engine --exportLayerInfo graph.json --shapes input:1x3x520x520
  
  # 动态引擎指定形状范围
  %(prog)s --loadEngine dynamic_model.engine --exportLayerInfo graph.json \\
           --minShapes input:1x3x320x320 \\
           --optShapes input:1x3x520x520 \\
           --maxShapes input:1x3x1024x1024
  
  # 自定义迭代次数
  %(prog)s --loadEngine model.engine --exportLayerInfo graph.json --iterations 500 --warmUp 20
        """
    )
    
    # 必需参数
    parser.add_argument('--loadEngine', type=str, required=True,
                       help='TensorRT引擎文件路径 (.engine)')
    parser.add_argument('--exportLayerInfo', type=str, required=True,
                       help='输出JSON图结构文件路径')
    
    # 形状参数 (用于动态引擎)
    shape_group = parser.add_argument_group('形状参数 (用于动态引擎)')
    shape_group.add_argument('--shapes', type=str,
                           help='指定输入形状，格式: "input_name:BxHxW" 或 "BxHxW"')
    
    shape_group.add_argument('--minShapes', type=str,
                           help='最小形状范围，格式同 --shapes')
    shape_group.add_argument('--optShapes', type=str,
                           help='最优形状范围，格式同 --shapes')
    shape_group.add_argument('--maxShapes', type=str,
                           help='最大形状范围，格式同 --shapes')
    
    # 性能测试参数
    perf_group = parser.add_argument_group('性能测试参数')
    perf_group.add_argument('--iterations', type=int,
                          help='迭代次数 (默认: 100)')
    perf_group.add_argument('--warmUp', type=int,
                          help='预热次数 (默认: 10)')
    
    # 其他参数
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细输出')
    
    args = parser.parse_args()
    
    # 打印标题
    print("=" * 60)
    print("TensorRT 引擎分析工具")
    print("=" * 60)
    
    # 检查必需参数
    if not os.path.exists(args.loadEngine):
        print(f"❌ 错误: 引擎文件不存在: {args.loadEngine}")
        sys.exit(1)
    
    # 分析引擎元数据
    file_size = analyze_engine_metadata(args.loadEngine)
    
    # 构建 trtexec 命令
    print(f"\n🔧 构建 trtexec 命令...")
    try:
        cmd = build_trtexec_command(args)
    except ValueError as e:
        print(f"❌ 参数错误: {e}")
        sys.exit(1)
    
    # 显示命令详情
    if args.verbose:
        print("命令参数:")
        for i, param in enumerate(cmd):
            if param.startswith("--"):
                print(f"  {param:20s} {cmd[i+1] if i+1 < len(cmd) else ''}")
    
    print(f"\n🚀 开始执行性能分析...")
    print("-" * 40)
    
    try:
        # 执行 trtexec 命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        # 输出结果
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("标准错误输出:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
        
        # 检查执行结果
        if result.returncode == 0:
            print("-" * 40)
            print(f"✅ 成功导出层信息到: {args.exportLayerInfo}")
            
            # 解析性能指标
            metrics = parse_trtexec_output(result.stdout)
            
            # 显示性能摘要
            print(f"\n📊 性能摘要:")
            if metrics['latency']:
                print(f"   平均延迟: {metrics['latency']:.2f} ms")
            if metrics['throughput']:
                print(f"   吞吐量: {metrics['throughput']:.1f} qps")
            if metrics['memory_usage']:
                print(f"   GPU内存峰值: {metrics['memory_usage']:.1f} MiB")
            
            # 检查生成的JSON文件
            if os.path.exists(args.exportLayerInfo):
                try:
                    with open(args.exportLayerInfo, 'r') as f:
                        graph_data = json.load(f)
                    layer_count = len(graph_data.get('layers', []))
                    print(f"   导出的层数: {layer_count}")
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"   JSON文件解析警告: {e}")
            
            print(f"✅ 分析完成!")
            
        else:
            print(f"❌ trtexec 执行失败，返回码: {result.returncode}")
            
            # 常见错误诊断
            if "could not be opened" in result.stderr:
                print("\n💡 可能的原因:")
                print("  1. 引擎文件损坏或格式不正确")
                print("  2. 引擎与当前TensorRT版本不兼容")
                print("  3. 文件权限问题")
            
            elif "input dimensions" in result.stderr.lower():
                print("\n💡 可能的原因:")
                print("  1. 动态引擎需要指定形状参数")
                print("  2. 指定的形状超出了引擎的优化范围")
                print("  3. 形状格式不正确")
                print("\n  尝试使用: --shapes input:1x3x520x520")
            
            sys.exit(1)
            
    except FileNotFoundError:
        print(f"❌ 错误: 未找到 trtexec 命令")
        print("请确保:")
        print("  1. TensorRT 已正确安装")
        print("  2. trtexec 在系统 PATH 中")
        print("  3. 或在 Docker 容器中已包含 TensorRT")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  分析被用户中断")
        sys.exit(130)
    
    except Exception as e:
        print(f"❌ 执行过程中发生未知错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()