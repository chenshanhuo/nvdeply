import tensorrt as trt
from pathlib import Path

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import os

import fnmatch
    


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
        onnx_path=None,
        engine_save_path=None,
        workspace=1 << 30,#默认参数就是1g，如果设置的话就用设置的，否则就用默认的
        dynamic=False,
        fp16=False,
        int8=False,
        calib_data_dir=None,
        calib_batch_size=1,
        calib_cacche="calib.cache",
        mixed_percision=False,
        fp16_layers=[],
        profiling_verbosity="DETAILED"

):
    '''
    这个函数是一个简化版本的ONNX转TensorRT引擎的函数，
    
    主要用于快速测试和调试。它接受一些基本参数，
    如ONNX文件路径、输出引擎路径、工作空间大小、是否启用FP16精度、是否启用动态输入形状以及输入的宽高等。
    函数内部会打印这些参数，并调用一个假设存在的`export_engine_simple`函数来执行实际的转换过程。
    新增int8量化相关参数 包括是否启用int8 以及校准数据目录 校准批次大小 校准缓存文件等
    '''
    print(f"ONNX Path: {onnx_path}")
    print(f"Engine Save Path: {engine_save_path}")
    if int8 and calib_data_dir is None:
        raise ValueError("启用 INT8 量化必须提供校准数据目录 (calib_data_dir)")
    #首先是trt相关对象的构建，比如说build，config logger
    #构建logger
    logger = trt.Logger(trt.Logger.INFO)# 创建TensorRT日志记录器，设置日志级别为INFO

    build = trt.Builder(logger) # 创建TensorRT构建器对象，用于构建引擎
    config = build.create_builder_config() # 创建构建配置对象，用于设置构建
    #config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED # 设置构建配置的性能分析详细程度为DETAILED，启用详细的性能分析日志输出

        # 设置详细分析级别 - 添加这里
    if profiling_verbosity == "DETAILED":
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        print("✅ 启用详细分析 (ProfilingVerbosity.DETAILED)")
    elif profiling_verbosity == "LAYER_NAMES_ONLY":
        config.profiling_verbosity = trt.ProfilingVerbosity.LAYER_NAMES_ONLY
        print("ℹ️  使用层名分析 (ProfilingVerbosity.LAYER_NAMES_ONLY)")
    elif profiling_verbosity == "NONE":
        config.profiling_verbosity = trt.ProfilingVerbosity.NONE
        print("⚠️  禁用分析 (ProfilingVerbosity.NONE)")
    else:
        # 默认使用详细分析
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        print("✅ 使用默认详细分析")


    #设置空间大小
    config.max_workspace_size = 1 << 30 # 设置构建过程中使用的最大工作空间大小为1GB 在这边就是设置的是字节数

    #下面使用trt的依赖库创建一个网络解析器，加载onnx模型，解析模型结构，构建引擎并保存到指定路径 也就是network和parser
    network = build.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) # 创建TensorRT网络定义对象，启用显式批处理模式
    parser = trt.OnnxParser(network, logger) # 使用network和logger创建ONNX解析器对象，用于解析ONNX模型
    with  open(onnx_path,"rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("无法解析ONNX文件",onnx_path) # 解析ONNX模型，如果解析失败则抛出异常
    
    #需要设置是否动态
    if dynamic:
        profile = build.create_optimization_profile()
        profile.set_shape("input", (1, 3, 520, 520), (1, 3, 520, 520), (1, 3, 520, 520)) # 设置动态输入形状的优化配置，指定输入名称和最小、最优、最大形状
        config.add_optimization_profile(profile) # 将优化配置添加到构建配置中

    if fp16 and build.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16) # 如果启用FP16精度且平台支持，则设置构建标志为FP16#全局设置
    elif int8 and build.platform_has_fast_fp16 and mixed_percision:
        config.set_flag(trt.BuilderFlag.FP16) # 如果启用混合精度且平台支持FP16，则设置构建标志为FP16#全局设置
        

        fp16_layers=fp16_layers or []
        for i in range(network.num_layers): # 遍历网络中的每一层
            layer = network.get_layer(i) # 获取网络中的每一层
            layer_name = layer.name # 获取层的名称
            layer_type = str(layer.type) # 获取层的类型

            should_use_fp16=(
                layer_name in fp16_layers or
                layer_type in fp16_layers or
                any(fnmatch.fnmatch(layer_name, pattern) for pattern in fp16_layers) or
                any(fnmatch.fnmatch(layer_type, pattern) for pattern in fp16_layers)
            )

            if layer.type ==trt.LayerType.CONSTANT and "Resize" in layer_type:
                continue
            if layer.type == trt.LayerType.SHUFFLE : # 对一些特殊的层比如说shuffle层 
                #进行特殊处理 因为有些shuffle层可能会导致fp16精度下降 但是又不想完全禁用fp16 所以就跳过这些层
                 continue
            
            if should_use_fp16:
                layer.precision = trt.DataType.HALF # 将层的计算精度设置为FP16
                layer.set_output_type(0, trt.DataType.HALF) # 将层的输出类型设置为FP16
                print(f"将层 {layer_name} ({layer_type}) 设置为 FP16") # 打印设置的信息
                for j in range(layer.num_outputs): # 对层的每个输出进行设置
                    layer.set_output_type(j, trt.DataType.HALF) # 将输出类型设置为FP16


        config.set_flag(trt.BuilderFlag.STRICT_TYPES) # 启用严格类型检查，确保层的计算精度和输出类型一致
        config.set_flag(trt.BuilderFlag.INT8)
        print("启用 INT8 量化") # 如果启用INT8且设备支持，则设置构建标志为INT8

        #生成校准器对象
        calibrator = FCNCalibrator(
                calibration_data_dir=calib_data_dir,
                batch_size=calib_batch_size,
                cache_file=calib_cacche
        )
        config.int8_calibrator = calibrator # 将校准器对象设置到构建配置中
        print(f"校准数据目录: {calib_data_dir}") # 打印校准数据目录
        print(f"校准批次大小: {calib_batch_size}") # 打印
        print(f"校准缓存文件: {calib_cacche}") # 打印校准缓存文件

    elif int8 and build.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("启用 INT8 量化") # 如果启用INT8且设备支持，则设置构建标志为INT8

        #生成校准器对象
        calibrator = FCNCalibrator(
                calibration_data_dir=calib_data_dir,
                batch_size=calib_batch_size,
                cache_file=calib_cacche
        )
        config.int8_calibrator = calibrator # 将校准器对象设置到构建配置中
        print(f"校准数据目录: {calib_data_dir}") # 打印校准数据目录
        print(f"校准批次大小: {calib_batch_size}") # 打印
        print(f"校准缓存文件: {calib_cacche}") # 打印校准缓存文件


    else:
        print("⚠️ 设备不支持 INT8 量化以及fp16，已降级为FP32") # 如果启用INT8但设备不支持，则打印警告信息
    

    #构建引擎并保存 fp16或者 fp32版本
    trt_version = int(trt.__version__.split(".")[0])
    if trt_version>=10:
        serialized_engine = build.build_serialized_network(network, config) # 构建序列化的TensorRT引擎
        with open(engine_save_path, "wb") as f:
            f.write(serialized_engine) # 将序列化的引擎写入文件
    else:
        engine = build.build_engine(network, config) # 构建TensorRT引擎
        if engine is None:
            raise RuntimeError("引擎构建失败") # 如果引擎构建失败则抛出异常
        with open(engine_save_path, "wb") as f:
            f.write(engine.serialize()) # 将引擎序列化并写入文件
    print(f"引擎已成功保存到: {engine_save_path}") # 打印保存路径




def main():
    '''
    首先先不使用参数化，先用变量代替
    '''
    onnx_path="deploy/NVDeploy/model/onnx/fcn-resnet50-12-marked_contain.onnx" #要转化的onnx
    engine_save_path=onnx_path.replace(".onnx","_int8_mixed.engine") #输出路径
    workspace=1 << 30 #工作空间大小
    dynamic=True
    fp16=False
    int8=True
    calib_data_dir="data/fcn-resnet50-12/calib" #校准数据目录
    calib_batch_size=1
    calib_cacche=onnx_path.replace(".onnx","_calib.cache") #校准缓存文件路径
    mixed_percision=False #是否启用混合精度
    fp16_layers = ["Conv_6","Conv_9"]

    if mixed_percision and not int8: #如果启用混合精度但没有启用int8 就启用fp16
        print("不要在非int8模式下启用混合精度，已自动启用FP16")


    if  int8 and fp16:raise ValueError("int8和fp16不能同时启用") # int8和fp16不能同时启用

    if len(fp16_layers)==0:
        print("没有指定需要使用FP16的层，将使用全局设置 int 8 ") # 如果没有指定需要使用FP16的层，则使用全局设置




    export_engine_simple(
        onnx_path=onnx_path,
        engine_save_path=engine_save_path,
        workspace=workspace,
        dynamic=dynamic,
        fp16=fp16,
        int8=int8,
        calib_data_dir=calib_data_dir,
        calib_batch_size=calib_batch_size,
        calib_cacche=calib_cacche,
        mixed_percision=mixed_percision,
        fp16_layers=fp16_layers,

    )



if __name__=="__main__":
    main()
    '''
    自己手写一个简单的onnx转trt的脚本，主要是为了调试和理解转换过程，代码结构和参数设计都比较简单，适合快速测试和验证。
    主要功能包括：
    在原先的基础上增加int8校准功能，

    '''