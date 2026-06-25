#!/usr/bin/env python3
"""
TensorRT 引擎转换工具 - 支持 FP32/FP16/INT8 量化及混合精度

功能：
- FP32/FP16/INT8 精度转换
- 混合精度：指定特定层使用 FP16
- 动态输入形状配置
- 详细性能分析 (ProfilingVerbosity)

示例：
  # FP16 转换
  python own_convert_int8.py --onnx model.onnx --fp16

  # INT8 量化
  python own_convert_int8.py --onnx model.onnx --int8 \
      --calib-data-dir ./calib --calib-batch-size 4

  # 混合精度 (INT8 + 部分层 FP16)
  python own_convert_int8.py --onnx model.onnx --int8 \
      --mixed-precision --fp16-layers "Conv_6,Conv_9,Relu_5"

  # 动态形状配置
  python own_convert_int8.py --onnx model.onnx --dynamic \
      --opt-shape 1,3,520,520 --max-shape 4,3,1040,1040
"""

import tensorrt as trt
from pathlib import Path
import argparse
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import os
import fnmatch
import time
    


class FCNCalibrator(trt.IInt8EntropyCalibrator2):
    '''
    FCN校准器 继承了trt的int8 交叉熵校准器 主要实现了三个方法 get_batch_size get_batch get_algorithm

    '''

    def  __init__(self,calibration_data_dir,batch_size=1,cache_file="calib.cache"):
        '''
        相关参数的初始化
        '''
        super(FCNCalibrator,self).__init__() # 调用父类的构造函数
        self.calib_data_dir = Path(calibration_data_dir) # 校准数据目录
        self.batch_size = batch_size # 批次大小
        self.cache_file = cache_file # 校准缓存文件

        self.calib_files=list(self.calib_data_dir.glob("*.npy")) # 获取校准数据目录下的所有npy文件

        #打乱文件顺序
        np.random.shuffle(self.calib_files)

        self.current_index=0# 当前索引

        #获取输入的形状 加一个文件夹非空的保护
        if len(self.calib_files)>0:
            sample_data=np.load(self.calib_files[0]) # 加载第一个npy
            self.input_shape=sample_data.shape # 获取输入的形状
            print(f"输入的形状为: {self.input_shape}") # 打印输入的形状
             
            # #计算每个样本的大小
            # self.sample_size = int(np.prod(self.input_shape)) * np.float32().nbytes

            # 计算每个样本的字节数
            # 注意：numpy数组的nbytes属性就是总字节数
            self.sample_size = sample_data.nbytes
            print(f"每个样本大小: {self.sample_size} 字节")

            # #在gpu上分配一个大的内存块 用于存放输入数据
            # self.device_input = np.empty(self.batch_size*self.sample_size,dytpe=np.float32) # 分配一个大的内存块 用于存放输入数据
            # #如何放置到gpu上？
            # self.device_input_ptr=cuda.men.alloc(self.device_input.nbytes) # 在GPU上分配内存
            # self.device_input_ptr = int(self.device_input_ptr) # 转换为整数地址
            
            #直接在gpu上分配
            self.device_input_ptr = cuda.mem_alloc(self.batch_size*self.sample_size) # 在GPU上分配内存 生成一个指正地址
            #self.device_input_ptr = int(self.device_input_ptr) # 转换为整数地址   #后续过程中的 getbatch方法要求传入的是整数地址
            #不可以在这边进行转换为整数地址 因为cuda.mem_alloc返回的是一个pycuda.driver.DeviceAllocation对象 
            # 直接转换为整数地址会导致后续的cuda.memcpy_htod方法无法识别这个地址 从而报错
            #所以在后续的get_batch方法中 直接传入self.device_input_ptr即可 
            # 因为cuda.memcpy_htod方法可以接受pycuda.driver.DeviceAllocation对象作为参数
        else:
            raise ValueError("校准数据目录下没有npy文件")


    def get_batch_size(self):
        return self.batch_size
    

    def get_batch(self,names):
        '''
        获得一个批次的数据 主要是从npy文件中加载数据，放到GPU上，并返回指针地址
        直到没有更多的数据了 就返回None
        '''
        if self.current_index >= len(self.calib_files): #是不是有可能会出现一定量的剩余数据不够校准？剩下的是否会浪费？
            return None # 没有更多的数据了 就返回None)
        
        #计算本批次的实际大小
        batch_end = min(self.current_index+self.batch_size, len(self.calib_files)) # 计算本批次的结束索引
        actual_batch_size = batch_end - self.current_index # 计算本批次的实际
        if actual_batch_size == 0:
            return None # 如果实际批次大小为0 就返回None
        batch_files = self.calib_files[self.current_index:batch_end] # 获取本批次的文件列表

        batch_data = []
        for i in range(actual_batch_size):
            data=np.load(batch_files[i]) # 加载npy文件
            if data.shape != self.input_shape:
                raise ValueError(f"输入数据的形状不匹配: {data.shape} != {self.input_shape}") # 如果输入数据的形状不匹配 就抛出异常
            batch_data.append(data) # 将数据添加到批次数据列表中

        #注意下面将类别的data使用concatenate连接成一个大的数组 以便于放到GPU上 因为原先的npy就是 nchw

        batch_data = np.concatenate(batch_data, axis=0).astype(np.float32) # 将批次数据连接成一个大的数组 并转换为float32类型

        #将数据放到GPU上
        cuda.memcpy_htod(self.device_input_ptr, batch_data) # 将数据从主机内存复制到GPU内存

        self.current_index=batch_end # 更新当前索引

        return [int(self.device_input_ptr)] # 返回指针地址 注意这里要求返回的是一个列表 因为可能有多个输入


    def read_calibration_cache(self):
        '''
        读取校准缓存 如果存在的话 就直接返回缓存内容 避免重复校准
        '''
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                cache = f.read()
                print(f"读取校准缓存成功: {self.cache_file}")
                return cache
        else:
            print(f"没有找到校准缓存: {self.cache_file}")
            return None

    def write_calibration_cache(self,cache):
        '''
        写入校准缓存 将校准结果写入文件 以便于下次使用
        '''
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            print(f"写入校准缓存成功: {self.cache_file}")

    



def export_engine_simple(
        onnx_path: str,
        engine_save_path: str,
        workspace: int = 1 << 30,
        dynamic: bool = False,
        min_shape: tuple = None,
        opt_shape: tuple = None,
        max_shape: tuple = None,
        fp16: bool = False,
        int8: bool = False,
        calib_data_dir: str = None,
        calib_batch_size: int = 1,
        calib_cache: str = "calib.cache",
        mixed_precision: bool = False,
        fp16_layers: list = None,
        profiling_verbosity: str = "DETAILED"
):
    """
    ONNX 转 TensorRT 引擎函数

    参数:
        onnx_path: ONNX 模型路径
        engine_save_path: 引擎保存路径
        workspace: 工作空间大小 (字节)
        dynamic: 是否启用动态形状
        min_shape: 动态形状最小值 (batch, channels, height, width)
        opt_shape: 动态形状最优值
        max_shape: 动态形状最大值
        fp16: 是否启用 FP16
        int8: 是否启用 INT8
        calib_data_dir: INT8 校准数据目录
        calib_batch_size: 校准批次大小
        calib_cache: 校准缓存文件路径
        mixed_precision: 是否启用混合精度
        fp16_layers: 混合精度模式下使用 FP16 的层列表
        profiling_verbosity: 性能分析级别 (DETAILED/LAYER_NAMES_ONLY/NONE)
    """
    fp16_layers = fp16_layers or []

    print(f"{'='*60}")
    print(f"TensorRT 引擎转换配置")
    print(f"{'='*60}")
    print(f"ONNX 路径: {onnx_path}")
    print(f"引擎路径: {engine_save_path}")
    print(f"工作空间: {workspace / (1<<30):.0f} GB")
    print(f"精度模式: ", end="")

    if int8 and mixed_precision:
        print("INT8 (混合精度)")
    elif int8:
        print("INT8")
    elif fp16:
        print("FP16")
    else:
        print("FP32")

    if int8 and calib_data_dir is None:
        raise ValueError("启用 INT8 量化必须提供校准数据目录 (--calib-data-dir)")

    # 构建 logger
    logger = trt.Logger(trt.Logger.INFO)
    build = trt.Builder(logger)
    config = build.create_builder_config()

    # 设置 profiling verbosity
    if profiling_verbosity == "DETAILED":
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        print("性能分析: DETAILED")
    elif profiling_verbosity == "LAYER_NAMES_ONLY":
        config.profiling_verbosity = trt.ProfilingVerbosity.LAYER_NAMES_ONLY
        print("性能分析: LAYER_NAMES_ONLY")
    elif profiling_verbosity == "NONE":
        config.profiling_verbosity = trt.ProfilingVerbosity.NONE
        print("性能分析: NONE")

    # 设置工作空间
    config.max_workspace_size = workspace

    # 创建网络和解析器
    network = build.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError(f"无法解析 ONNX 文件: {onnx_path}")

    # 打印输入输出信息
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    print(f"\n模型信息:")
    for inp in inputs:
        print(f"  输入: {inp.name}, 形状: {inp.shape}")
    for out in outputs:
        print(f"  输出: {out.name}")

    # 设置动态形状
    if dynamic:
        profile = build.create_optimization_profile()
        input_name = inputs[0].name if inputs else "input"

        # 默认值处理
        if min_shape is None:
            min_shape = (1, 3, 260, 260)
        if opt_shape is None:
            opt_shape = (1, 3, 520, 520)
        if max_shape is None:
            max_shape = (1, 3, 1040, 1040)

        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        print(f"\n动态形状配置:")
        print(f"  最小: {min_shape}")
        print(f"  最优: {opt_shape}")
        print(f"  最大: {max_shape}")

    if fp16 and build.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("启用 FP16 精度")
    elif int8 and build.platform_has_fast_fp16 and mixed_precision:
        config.set_flag(trt.BuilderFlag.FP16)

        print("启用混合精度模式 (INT8 + 部分层 FP16)")
        print(f"FP16 层: {fp16_layers}")

        for i in range(network.num_layers):
            layer = network.get_layer(i)
            layer_name = layer.name
            layer_type = str(layer.type)

            should_use_fp16 = (
                layer_name in fp16_layers or
                layer_type in fp16_layers or
                any(fnmatch.fnmatch(layer_name, pattern) for pattern in fp16_layers) or
                any(fnmatch.fnmatch(layer_type, pattern) for pattern in fp16_layers)
            )

            # 跳过可能导致精度问题的特殊层
            if layer.type == trt.LayerType.CONSTANT and "Resize" in layer_type:
                continue
            if layer.type == trt.LayerType.SHUFFLE:
                continue

            if should_use_fp16:
                layer.precision = trt.DataType.HALF
                for j in range(layer.num_outputs):
                    layer.set_output_type(j, trt.DataType.HALF)
                print(f"  [FP16] {layer_name} ({layer_type})")

        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        config.set_flag(trt.BuilderFlag.INT8)

        # 创建校准器
        calibrator = FCNCalibrator(
            calibration_data_dir=calib_data_dir,
            batch_size=calib_batch_size,
            cache_file=calib_cache
        )
        config.int8_calibrator = calibrator
        print(f"校准数据: {calib_data_dir}")
        print(f"校准批次: {calib_batch_size}")

    elif int8 and build.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("启用 INT8 量化")

        # 创建校准器
        calibrator = FCNCalibrator(
            calibration_data_dir=calib_data_dir,
            batch_size=calib_batch_size,
            cache_file=calib_cache
        )
        config.int8_calibrator = calibrator
        print(f"校准数据: {calib_data_dir}")
        print(f"校准批次: {calib_batch_size}")

    else:
        print("设备不支持 INT8/FP16，使用 FP32")

    # 构建引擎
    print(f"\n开始构建引擎...")
    start_time = time.time()

    trt_version = int(trt.__version__.split(".")[0])
    try:
        if trt_version >= 10:
            serialized_engine = build.build_serialized_network(network, config)
        else:
            engine = build.build_engine(network, config)
            if engine is None:
                raise RuntimeError("引擎构建失败")
            serialized_engine = engine.serialize()

        with open(engine_save_path, "wb") as f:
            f.write(serialized_engine)

        elapsed = time.time() - start_time
        size_mb = os.path.getsize(engine_save_path) / 1024 / 1024

        print(f"\n{'='*60}")
        print(f"引擎构建成功!")
        print(f"  保存路径: {engine_save_path}")
        print(f"  文件大小: {size_mb:.2f} MB")
        print(f"  构建时间: {elapsed:.2f} 秒")
        print(f"{'='*60}")

    except Exception as e:
        print(f"引擎构建失败: {e}")
        raise




def parse_shape(shape_str: str) -> tuple:##将输入的形状字符串解析为元组
    """解析形状字符串 'batch,channels,height,width' -> tuple"""
    parts = [int(x.strip()) for x in shape_str.split(',')]
    if len(parts) != 4:
        raise ValueError(f"形状必须包含4个维度: batch,channels,height,width")
    return tuple(parts)


def main():
    # 部分示例，用于提示
    parser = argparse.ArgumentParser(
        description="TensorRT 引擎转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        示例:
        # FP16 转换
        --onnx model.onnx --fp16

        # INT8 量化
        --onnx model.onnx --int8 --calib-data-dir ./calib

        # 混合精度
        --onnx model.onnx --int8 --mixed-precision --fp16-layers "Conv_6,Relu_5"

        # 动态形状
        --onnx model.onnx --dynamic --opt-shape 1,3,520,520 --max-shape 4,3,1040,1040
        """
    )

    # 必需参数
    parser.add_argument('--onnx', '-i', required=True, help='输入 ONNX 模型路径') #必须有
    parser.add_argument('--output', '-o', default=None, help='输出引擎文件路径 (默认: 自动生成)') #如果没有生成

    # 精度设置
    precision_group = parser.add_mutually_exclusive_group()
    precision_group.add_argument('--fp16', action='store_true', help='启用 FP16 精度')
    precision_group.add_argument('--int8', action='store_true', help='启用 INT8 量化')

    # 混合精度 (仅 INT8 模式有效)
    parser.add_argument('--mixed-precision', action='store_true',
                        help='启用混合精度 (INT8 + 部分层 FP16)')
    parser.add_argument('--fp16-layers', type=str, default='',
                        help='混合精度模式下使用 FP16 的层名称 (逗号分隔，支持通配符)')

    # INT8 校准参数
    parser.add_argument('--calib-data-dir', default=None,
                        help='INT8 校准数据目录 (包含 .npy 文件)')
    parser.add_argument('--calib-batch-size', type=int, default=1,
                        help='校准批次大小 (默认: 1)')
    parser.add_argument('--calib-cache', default=None,
                        help='校准缓存文件路径 (默认: <onnx>.cache)')

    # 动态形状
    parser.add_argument('--dynamic', action='store_true', help='启用动态输入形状')
    parser.add_argument('--min-shape', type=str, default=None,
                        help='动态形状最小值: batch,channels,height,width')
    parser.add_argument('--opt-shape', type=str, default=None,
                        help='动态形状最优值: batch,channels,height,width')
    parser.add_argument('--max-shape', type=str, default=None,
                        help='动态形状最大值: batch,channels,height,width')

    # 其他参数
    parser.add_argument('--workspace', type=int, default=4096,
                        help='工作空间大小 MB (默认: 4096)')
    parser.add_argument('--profiling', choices=['DETAILED', 'LAYER_NAMES_ONLY', 'NONE'],
                        default='DETAILED', help='性能分析级别 (默认: DETAILED)')
    parser.add_argument('--verbose', action='store_true', help='显示详细日志')

    args = parser.parse_args()

    # 处理输出路径 如果没有指定输出路径，则根据输入 ONNX 文件生成默认输出路径
    if args.output is None:
        onnx_path = Path(args.onnx)
        onnx_stem = onnx_path.stem  # 去除扩展名的文件名
        output_dir = onnx_path.parent  # 保留原目录
        if args.fp16:
            output_name = f"{onnx_stem}_fp16.engine"
        elif args.mixed_precision:
            output_name = f"{onnx_stem}_int8_mixed.engine"
        else:
            output_name = f"{onnx_stem}_int8.engine"
        args.output = str(output_dir / output_name)

    # 处理校准缓存路径
    if args.calib_cache is None: #如果没有指定校准缓存路径，则根据输入 ONNX 文件生成默认输出路径
        onnx_path = Path(args.onnx)
        args.calib_cache = str(onnx_path.parent / f"{onnx_path.stem}.cache")

    # 解析形状
    min_shape = parse_shape(args.min_shape) if args.min_shape else None
    opt_shape = parse_shape(args.opt_shape) if args.opt_shape else None
    max_shape = parse_shape(args.max_shape) if args.max_shape else None

    # 解析 FP16 层列表
    fp16_layers = [l.strip() for l in args.fp16_layers.split(',') if l.strip()]

    # 验证
    if args.mixed_precision and not args.int8:
        print("警告: --mixed-precision 仅在 INT8 模式下生效，将启用 INT8")
        args.int8 = True

    if args.int8 and not args.calib_data_dir: #如果没有指定校准数据目录，则报错
        parser.error("--int8 模式需要 --calib-data-dir 参数")

    # 调用转换函数
    export_engine_simple(
        onnx_path=args.onnx,
        engine_save_path=args.output,
        workspace=args.workspace * 1024 * 1024,
        dynamic=args.dynamic,
        min_shape=min_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
        fp16=args.fp16,
        int8=args.int8,
        calib_data_dir=args.calib_data_dir,
        calib_batch_size=args.calib_batch_size,
        calib_cache=args.calib_cache,
        mixed_precision=args.mixed_precision,
        fp16_layers=fp16_layers,
        profiling_verbosity=args.profiling
    )


if __name__ == "__main__":
    main()