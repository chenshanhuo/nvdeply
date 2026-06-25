import tensorrt as trt
from pathlib import Path
import json
import argparse
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import os
import time


def export_engine_simple(
    onnx_file: str,
    engine_file: str = None,
    workspace: int = 1 << 30,
    fp16: bool = False,
    dynamic: bool = False,
    shape: tuple = (1, 3, 520, 520),
    calib_data_dir: str = None,
    verbose: bool = False,
) -> None:
    """简化的 ONNX 转 TensorRT 引擎函数，专注于 FCN-ResNet50 转换"""
    
    # 设置输出文件名
    if engine_file is None:
        engine_file = str(Path(onnx_file).with_suffix(".engine"))
    
    print(f"转换模型: {onnx_file}")
    print(f"输出引擎: {engine_file}")
    print(f"精度模式: {'FP16' if fp16 else 'FP32'}")
    print(f"动态输入: {dynamic}")
    print(f"输入形状: {shape}")
    
    # 创建日志记录器
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE
    
    # 创建 builder 和 config
    builder = trt.Builder(logger)
    config = builder.create_builder_config()  
    
    # 设置 workspace
    config.max_workspace_size = workspace
    
    # 创建网络
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # 加载 ONNX 模型
    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError(f"无法解析 ONNX 文件: {onnx_file}")
    
    # 打印输入输出信息
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    
    print("\n模型输入输出信息:")
    for inp in inputs:
        print(f"  输入: {inp.name}, 形状: {inp.shape}, 类型: {inp.dtype}")
    for out in outputs:
        print(f"  输出: {out.name}, 形状: {out.shape}, 类型: {out.dtype}")
    
    # 设置动态形状（如果启用）
    if dynamic:
        profile = builder.create_optimization_profile()
        for inp in inputs:
            # 使用用户指定的形状作为最优形状
            profile.set_shape(
                inp.name, 
                min=(1, 3, shape[2]//2, shape[3]//2),  # 最小尺寸
                opt=shape,                              # 最优尺寸
                max=(4, 3, shape[2]*2, shape[3]*2)      # 最大尺寸
            )
        config.add_optimization_profile(profile)
        print(f"动态形状范围: 最小{shape[2]//2}x{shape[3]//2}, 最优{shape[2]}x{shape[3]}, 最大{shape[2]*2}x{shape[3]*2}")
    
    # 设置 FP16 标志（让 TensorRT 自动处理）
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("启用 FP16 精度")
    elif fp16 and not builder.platform_has_fast_fp16:
        print("警告: 当前平台不支持 FP16，将使用 FP32")
    
    # 构建引擎
    print(f"\n开始构建引擎...")
    start_time = time.time()
    
    try:
        # TensorRT 版本兼容性处理
        trt_version = int(trt.__version__.split(".")[0])
        if trt_version >= 10:
            # TensorRT 10.x
            serialized_engine = builder.build_serialized_network(network, config)
        else:
            # TensorRT 8.x/9.x
            engine = builder.build_engine(network, config)
            if engine is None:
                raise RuntimeError("引擎构建失败")
            serialized_engine = engine.serialize()
        
        # 保存引擎文件
        with open(engine_file, "wb") as f:
            f.write(serialized_engine)
        
        elapsed_time = time.time() - start_time
        print(f"✅ 引擎构建成功!")
        print(f"   保存到: {engine_file}")
        print(f"   文件大小: {os.path.getsize(engine_file)/1024/1024:.2f} MB")
        print(f"   构建时间: {elapsed_time:.2f} 秒")
        
    except Exception as e:
        print(f"❌ 引擎构建失败: {e}")
        raise


def main():
    """主函数：解析参数并调用转换函数"""
    parser = argparse.ArgumentParser(description="ONNX 转 TensorRT 引擎 (简化版)")
    
    # 必需参数
    parser.add_argument("--onnx", type=str, required=True, help="输入 ONNX 模型路径") # 要转化的onnx
    parser.add_argument("--output", type=str, default=None, help="输出引擎文件路径 (可选)") #输出路径
    
    # 精度和性能参数
    parser.add_argument("--fp16", action="store_true", help="启用 FP16 精度")
    parser.add_argument("--workspace", type=int, default=2048, help="工作空间大小 (MB)")
    
    # 输入形状参数
    parser.add_argument("--dynamic", action="store_true", help="启用动态输入形状")
    parser.add_argument("--height", type=int, default=520, help="输入高度")
    parser.add_argument("--width", type=int, default=520, help="输入宽度")
    
    # 其他参数
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    
    args = parser.parse_args()
    
    # 打印参数
    print("=" * 60)
    print("ONNX 转 TensorRT 引擎 (简化版)")
    print("=" * 60)
    print(f"ONNX 文件: {args.onnx}")
    print(f"输出文件: {args.output or '自动生成'}")
    print(f"精度模式: {'FP16' if args.fp16 else 'FP32'}")
    print(f"动态输入: {args.dynamic}")
    print(f"输入尺寸: {args.height}x{args.width}")
    print(f"工作空间: {args.workspace} MB")
    print("=" * 60)
    
    # 检查文件是否存在
    if not os.path.exists(args.onnx):
        print(f"错误: ONNX 文件不存在: {args.onnx}")
        return
    
    # 准备参数
    shape = (1, 3, args.height, args.width) # 
    workspace_mb = args.workspace * 1024 * 1024  # 转换为字节
    
    # 执行转换
    try:
        export_engine_simple(
            onnx_file=args.onnx,
            engine_file=args.output,
            workspace=workspace_mb,
            fp16=args.fp16,
            dynamic=args.dynamic,
            shape=shape,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"\n转换过程中发生错误:")
        print(f"  {type(e).__name__}: {e}")
        
        # 提供调试建议
        print(f"\n调试建议:")
        print("1. 检查 ONNX 模型是否有效: python -c \"import onnx; model = onnx.load('model.onnx'); onnx.checker.check_model(model)\"")
        print("2. 尝试不启用 FP16: 移除 --fp16 参数")
        print("3. 尝试固定输入尺寸: 移除 --dynamic 参数")
        print("4. 增加工作空间: 使用 --workspace 4096 或更高")
        print("5. 检查 TensorRT 和 CUDA 版本兼容性")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())