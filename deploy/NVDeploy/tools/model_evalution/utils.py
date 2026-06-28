"""
工具函数模块
"""
import json
import subprocess
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class PrecisionType(Enum):
    """精度类型枚举"""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class LayerProfile:
    """层性能数据"""
    name: str
    layer_type: str
    precision: str
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    percentage: float
    input_shapes: List[List[int]] = field(default_factory=list)
    output_shapes: List[List[int]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def flops_estimate(self) -> int:
        """估算该层的FLOPs"""
        return estimate_layer_flops(
            self.layer_type, 
            self.input_shapes, 
            self.output_shapes,
            self.metadata
        )


@dataclass
class ModelProfile:
    """模型整体性能数据"""
    model_name: str
    engine_path: str
    total_time_ms: float
    throughput: float  # FPS
    latency_mean_ms: float
    latency_median_ms: float
    latency_99_ms: float
    gpu_compute_time_ms: float
    host_walltime_ms: float
    layers: List[LayerProfile]
    memory_usage_mb: float = 0.0
    precision: str = "unknown"
    
    @property
    def total_flops(self) -> int:
        """模型总FLOPs估算"""
        return sum(layer.flops_estimate for layer in self.layers)
    
    @property
    def actual_flops(self) -> float:
        """实际FLOPS (每秒浮点运算次数)"""
        if self.gpu_compute_time_ms > 0:
            return self.total_flops / (self.gpu_compute_time_ms / 1000.0)
        return 0.0


def parse_dimension_string(dim_str: str) -> List[int]:
    """
    解析维度字符串，如 "1x3x224x224" 或 "(1, 3, 224, 224)" 或 "[1,3,224,224]"
    """
    if not dim_str:
        return []
    
    # 移除括号和空格
    dim_str = dim_str.strip().replace('(', '').replace(')', '').replace('[', '').replace(']', '')
    
    # 尝试不同的分隔符
    for sep in ['x', ',', ' ']:
        if sep in dim_str:
            parts = [p.strip() for p in dim_str.split(sep) if p.strip()]
            try:
                return [int(p) for p in parts if p.isdigit() or (p.startswith('-') and p[1:].isdigit())]
            except ValueError:
                continue
    
    # 单个数字
    try:
        return [int(dim_str)]
    except ValueError:
        return []


def extract_shapes_from_name(layer_name: str) -> Tuple[List[List[int]], List[List[int]]]:
    """
    从层名称中提取形状信息
    trtexec的层名称通常包含形状信息，如:
    "Conv_0 input[0]: (1,3,224,224) output[0]: (1,64,112,112)"
    """
    input_shapes = []
    output_shapes = []
    
    # 匹配 input[n]: (dims) 或 output[n]: (dims) 格式
    input_pattern = r'input\[\d+\]:\s*\(([^)]+)\)'
    output_pattern = r'output\[\d+\]:\s*\(([^)]+)\)'
    
    for match in re.finditer(input_pattern, layer_name, re.IGNORECASE):
        dims = parse_dimension_string(match.group(1))
        if dims:
            input_shapes.append(dims)
    
    for match in re.finditer(output_pattern, layer_name, re.IGNORECASE):
        dims = parse_dimension_string(match.group(1))
        if dims:
            output_shapes.append(dims)
    
    # 尝试匹配其他常见格式 [NxCxHxW]
    if not input_shapes and not output_shapes:
        dim_pattern = r'\[(\d+x\d+(?:x\d+)*)\]'
        matches = re.findall(dim_pattern, layer_name)
        for i, match in enumerate(matches):
            dims = parse_dimension_string(match)
            if dims:
                if i == 0:
                    input_shapes.append(dims)
                else:
                    output_shapes.append(dims)
    
    return input_shapes, output_shapes


def estimate_layer_flops(layer_type: str, input_shapes: List[List[int]], 
                         output_shapes: List[List[int]],
                         metadata: Dict[str, Any] = None) -> int:
    """
    估算单层FLOPs
    
    常见层类型的FLOPs计算:
    - Conv2D: 2 * K_h * K_w * C_in * C_out * H_out * W_out
    - MatMul/FC: 2 * M * N * K
    - BatchNorm: 4 * elements
    - ReLU/激活: elements
    """
    if metadata is None:
        metadata = {}
    
    layer_type_lower = layer_type.lower()
    
    try:
        # 卷积层
        if any(x in layer_type_lower for x in ['conv', 'convolution']):
            if output_shapes and len(output_shapes) > 0 and len(output_shapes[0]) >= 4:
                n, c_out, h_out, w_out = output_shapes[0][:4]
                c_in = input_shapes[0][1] if input_shapes and len(input_shapes[0]) >= 2 else c_out
                
                # 从metadata获取kernel size，默认3x3
                k_h = metadata.get('kernel_h', 3)
                k_w = metadata.get('kernel_w', 3)
                
                # 检查是否是depthwise conv
                groups = metadata.get('groups', 1)
                if groups == c_in and groups == c_out:
                    # Depthwise conv
                    return 2 * k_h * k_w * c_out * h_out * w_out * n
                else:
                    # Standard conv
                    return 2 * k_h * k_w * (c_in // max(groups, 1)) * c_out * h_out * w_out * n
            
            # 如果只有input shapes
            elif input_shapes and len(input_shapes) > 0 and len(input_shapes[0]) >= 4:
                n, c_in, h_in, w_in = input_shapes[0][:4]
                c_out = metadata.get('out_channels', c_in)
                k_h = metadata.get('kernel_h', 3)
                k_w = metadata.get('kernel_w', 3)
                return 2 * k_h * k_w * c_in * c_out * h_in * w_in * n
                    
        # 矩阵乘法 / 全连接层
        elif any(x in layer_type_lower for x in ['matmul', 'gemm', 'fc', 'linear', 'fullyconnected', 'innerproduct']):
            if len(input_shapes) >= 2:
                shape_a = input_shapes[0]
                shape_b = input_shapes[1]
                if len(shape_a) >= 2 and len(shape_b) >= 2:
                    m = shape_a[-2] if len(shape_a) > 1 else 1
                    k = shape_a[-1]
                    n = shape_b[-1]
                    batch = 1
                    for dim in shape_a[:-2]:
                        batch *= dim
                    return 2 * batch * m * n * k
            elif len(input_shapes) == 1 and output_shapes:
                in_features = input_shapes[0][-1] if input_shapes[0] else 0
                out_features = output_shapes[0][-1] if output_shapes[0] else 0
                batch = 1
                for dim in input_shapes[0][:-1]:
                    batch *= dim
                if in_features and out_features:
                    return 2 * batch * in_features * out_features
                    
        # 归一化层 (BatchNorm, LayerNorm, InstanceNorm)
        elif any(x in layer_type_lower for x in ['norm', 'scale', 'batchnorm', 'layernorm']):
            shapes = input_shapes if input_shapes else output_shapes
            if shapes and len(shapes) > 0:
                elements = 1
                for dim in shapes[0]:
                    elements *= max(dim, 1)
                return 4 * elements
                
        # 激活函数
        elif any(act in layer_type_lower for act in ['relu', 'sigmoid', 'tanh', 'gelu', 'swish', 'silu', 'activation', 'leakyrelu']):
            shapes = input_shapes if input_shapes else output_shapes
            if shapes and len(shapes) > 0:
                elements = 1
                for dim in shapes[0]:
                    elements *= max(dim, 1)
                return elements
                
        # 池化层
        elif any(x in layer_type_lower for x in ['pool', 'pooling']):
            shapes = output_shapes if output_shapes else input_shapes
            if shapes and len(shapes) > 0:
                elements = 1
                for dim in shapes[0]:
                    elements *= max(dim, 1)
                kernel_size = metadata.get('kernel_size', 9)
                return elements * kernel_size
                
        # 注意力机制 / Softmax
        elif any(x in layer_type_lower for x in ['attention', 'softmax', 'multihead']):
            shapes = input_shapes if input_shapes else output_shapes
            if shapes and len(shapes) > 0:
                elements = 1
                for dim in shapes[0]:
                    elements *= max(dim, 1)
                return 5 * elements
        
        # Elementwise操作 (Add, Mul, etc.)
        elif any(x in layer_type_lower for x in ['add', 'sum', 'mul', 'elementwise', 'eltwise']):
            shapes = output_shapes if output_shapes else input_shapes
            if shapes and len(shapes) > 0:
                elements = 1
                for dim in shapes[0]:
                    elements *= max(dim, 1)
                return elements
        
        # Reshape/Transpose等不计算的层
        elif any(x in layer_type_lower for x in ['reshape', 'transpose', 'permute', 'flatten', 'squeeze', 'unsqueeze', 'concat', 'slice', 'shuffle', 'reformat']):
            return 0
                
    except Exception as e:
        pass
    
    # 默认返回基于输出大小的估算
    shapes = output_shapes if output_shapes else input_shapes
    if shapes and len(shapes) > 0:
        elements = 1
        for dim in shapes[0]:
            elements *= max(dim, 1)
        return elements
    
    return 0


def _extract_precision_from_format(format_str: str) -> str:
    """从Format/Datatype字符串中提取精度"""
    format_upper = format_str.upper()
    
    if 'FP32' in format_upper or 'FLOAT32' in format_upper:
        return 'FP32'
    elif 'FLOAT' in format_upper and 'FP16' not in format_upper and 'HALF' not in format_upper:
        return 'FP32'
    elif 'FP16' in format_upper or 'HALF' in format_upper or 'FLOAT16' in format_upper:
        return 'FP16'
    elif 'INT8' in format_upper:
        return 'INT8'
    elif 'INT32' in format_upper:
        return 'INT32'
    elif 'INT4' in format_upper:
        return 'INT4'
    elif 'BOOL' in format_upper:
        return 'BOOL'
    
    return 'UNKNOWN'


def _normalize_precision(precision: str) -> str:
    """规范化精度字符串"""
    if not precision:
        return 'UNKNOWN'
    
    precision = precision.upper().strip()
    
    precision_map = {
        'FLOAT': 'FP32',
        'FLOAT32': 'FP32',
        'FP32': 'FP32',
        'HALF': 'FP16',
        'FLOAT16': 'FP16',
        'FP16': 'FP16',
        'INT8': 'INT8',
        'INT32': 'INT32',
        'INT4': 'INT4',
        'BOOL': 'BOOL',
        'UNKNOWN': 'UNKNOWN',
    }
    
    if precision in precision_map:
        return precision_map[precision]
    
    for key, value in precision_map.items():
        if key in precision:
            return value
    
    return precision if precision else 'UNKNOWN'


def _infer_layer_type_from_name(name: str) -> str:
    """从层名称推断层类型"""
    name_lower = name.lower()
    
    type_keywords = {
        'conv': 'Convolution',
        'bn': 'BatchNorm',
        'batchnorm': 'BatchNorm',
        'relu': 'ReLU',
        'pool': 'Pooling',
        'fc': 'FullyConnected',
        'linear': 'Linear',
        'matmul': 'MatMul',
        'gemm': 'GEMM',
        'add': 'Add',
        'concat': 'Concat',
        'softmax': 'Softmax',
        'sigmoid': 'Sigmoid',
        'reshape': 'Reshape',
        'transpose': 'Transpose',
        'slice': 'Slice',
        'attention': 'Attention',
        'layernorm': 'LayerNorm',
        'gelu': 'GELU',
        'silu': 'SiLU',
        'scale': 'Scale',
        'elementwise': 'Elementwise',
        'shuffle': 'Shuffle',
        'reformat': 'Reformat',
    }
    
    for keyword, layer_type in type_keywords.items():
        if keyword in name_lower:
            return layer_type
    
    return 'Unknown'


def _build_layer_info_map(layer_info_data) -> Dict[str, Dict]:
    """构建层名称到层信息的映射表"""
    layer_info_map = {}
    
    try:
        if isinstance(layer_info_data, dict):
            if 'Layers' in layer_info_data:
                layers_list = layer_info_data['Layers']
                if isinstance(layers_list, list):
                    for layer in layers_list:
                        if isinstance(layer, dict):
                            name = layer.get('Name', layer.get('name', ''))
                            if name:
                                layer_info_map[name] = layer
            elif 'layers' in layer_info_data:
                layers_list = layer_info_data['layers']
                if isinstance(layers_list, list):
                    for layer in layers_list:
                        if isinstance(layer, dict):
                            name = layer.get('Name', layer.get('name', ''))
                            if name:
                                layer_info_map[name] = layer
            else:
                for name, info in layer_info_data.items():
                    if isinstance(info, dict):
                        layer_info_map[name] = info
                        
        elif isinstance(layer_info_data, list):
            for layer in layer_info_data:
                if isinstance(layer, dict):
                    name = layer.get('Name', layer.get('name', ''))
                    if name:
                        layer_info_map[name] = layer
                    
    except Exception as e:
        print(f"构建层信息映射表时出错: {e}")
    
    return layer_info_map


def _extract_shapes(layer_data: dict, layer_name: str) -> Tuple[List[List[int]], List[List[int]]]:
    """从层数据中提取输入输出形状"""
    input_shapes = []
    output_shapes = []
    
    # 方法1: trtexec exportLayerInfo 格式
    if 'Inputs' in layer_data:
        inputs = layer_data['Inputs']
        if isinstance(inputs, list):
            for inp in inputs:
                if isinstance(inp, dict) and 'Dimensions' in inp:
                    dims = inp['Dimensions']
                    if isinstance(dims, list):
                        input_shapes.append([int(d) for d in dims])
    
    if 'Outputs' in layer_data:
        outputs = layer_data['Outputs']
        if isinstance(outputs, list):
            for out in outputs:
                if isinstance(out, dict) and 'Dimensions' in out:
                    dims = out['Dimensions']
                    if isinstance(dims, list):
                        output_shapes.append([int(d) for d in dims])
    
    # 方法2: 小写键名
    if not input_shapes and 'inputs' in layer_data:
        inputs = layer_data['inputs']
        if isinstance(inputs, list):
            for inp in inputs:
                if isinstance(inp, dict):
                    dims = inp.get('Dimensions', inp.get('dimensions', inp.get('shape', [])))
                    if dims:
                        input_shapes.append(_ensure_int_list(dims))
    
    if not output_shapes and 'outputs' in layer_data:
        outputs = layer_data['outputs']
        if isinstance(outputs, list):
            for out in outputs:
                if isinstance(out, dict):
                    dims = out.get('Dimensions', out.get('dimensions', out.get('shape', [])))
                    if dims:
                        output_shapes.append(_ensure_int_list(dims))
    
    # 方法3: inputShapes/outputShapes 格式
    if not input_shapes:
        for key in ['inputShapes', 'InputShapes', 'input_shapes']:
            if key in layer_data:
                input_shapes = _parse_shapes_field(layer_data[key])
                break
    
    if not output_shapes:
        for key in ['outputShapes', 'OutputShapes', 'output_shapes']:
            if key in layer_data:
                output_shapes = _parse_shapes_field(layer_data[key])
                break
    
    # 方法4: 从层名称中提取
    if not input_shapes and not output_shapes:
        input_shapes, output_shapes = extract_shapes_from_name(layer_name)
    
    return input_shapes, output_shapes


def _ensure_int_list(dims) -> List[int]:
    """确保返回整数列表"""
    if isinstance(dims, str):
        return parse_dimension_string(dims)
    
    result = []
    for d in dims:
        try:
            result.append(int(d))
        except (ValueError, TypeError):
            pass
    return result


def _parse_shapes_field(shapes) -> List[List[int]]:
    """解析shapes字段"""
    result = []
    
    if not shapes:
        return result
    
    if isinstance(shapes, list):
        for shape in shapes:
            if isinstance(shape, dict):
                dims = shape.get('dimensions', shape.get('shape', shape.get('dims', [])))
                if dims:
                    result.append(_ensure_int_list(dims))
            elif isinstance(shape, list):
                result.append(_ensure_int_list(shape))
            elif isinstance(shape, str):
                dims = parse_dimension_string(shape)
                if dims:
                    result.append(dims)
    elif isinstance(shapes, str):
        dims = parse_dimension_string(shapes)
        if dims:
            result.append(dims)
    
    return result


def _extract_metadata(layer_data: dict) -> Dict[str, Any]:
    """提取用于FLOPs计算的元数据"""
    metadata = {}
    
    # Kernel size
    if 'Kernel' in layer_data:
        kernel = layer_data['Kernel']
        if isinstance(kernel, list) and len(kernel) >= 2:
            metadata['kernel_h'] = int(kernel[0])
            metadata['kernel_w'] = int(kernel[1])
    else:
        for key in ['kernelSize', 'kernel_size', 'kernel', 'ksize']:
            if key in layer_data:
                val = layer_data[key]
                if isinstance(val, list) and len(val) >= 2:
                    metadata['kernel_h'] = int(val[0])
                    metadata['kernel_w'] = int(val[1])
                elif isinstance(val, int):
                    metadata['kernel_h'] = val
                    metadata['kernel_w'] = val
                break
    
    # Groups
    if 'Groups' in layer_data:
        metadata['groups'] = int(layer_data['Groups'])
    else:
        for key in ['groups', 'group', 'numGroups']:
            if key in layer_data:
                metadata['groups'] = int(layer_data[key])
                break
    
    # Output channels
    if 'OutMaps' in layer_data:
        metadata['out_channels'] = int(layer_data['OutMaps'])
    else:
        for key in ['outChannels', 'out_channels', 'numOutputs']:
            if key in layer_data:
                metadata['out_channels'] = int(layer_data[key])
                break
    
    # Stride
    if 'Stride' in layer_data:
        stride = layer_data['Stride']
        if isinstance(stride, list) and len(stride) >= 2:
            metadata['stride_h'] = int(stride[0])
            metadata['stride_w'] = int(stride[1])
    
    # ParameterType
    if 'ParameterType' in layer_data:
        metadata['parameter_type'] = layer_data['ParameterType']
    
    return metadata


def _parse_layer_info(layer_data: dict, layer_info_map: Dict[str, Dict] = None) -> Optional[LayerProfile]:
    """解析单层信息"""
    if layer_info_map is None:
        layer_info_map = {}
    
    try:
        # 获取层名称
        name = None
        for key in ['name', 'Name', 'layerName', 'LayerName']:
            if key in layer_data:
                name = layer_data[key]
                break
        if name is None:
            name = 'unknown'
        
        # 从layer_info_map中获取补充信息
        extra_info = layer_info_map.get(name, {})
        if not isinstance(extra_info, dict):
            extra_info = {}
        
        # 合并数据
        merged_data = {**layer_data, **extra_info}
        
        # 获取层类型
        layer_type = None
        for key in ['ParameterType', 'parameterType', 'LayerType', 'layerType', 'type', 'Type', 'kind']:
            if key in merged_data:
                layer_type = merged_data[key]
                break
        
        if not layer_type:
            layer_type = _infer_layer_type_from_name(name)
        
        # 获取时间信息
        avg_time = 0.0
        for key in ['averageMs', 'AverageMs', 'timeMs', 'TimeMs', 'time', 'latency', 'ms', 'duration']:
            if key in merged_data:
                val = merged_data[key]
                if isinstance(val, dict):
                    avg_time = val.get('mean', val.get('avg', val.get('average', 0)))
                else:
                    try:
                        avg_time = float(val)
                    except (ValueError, TypeError):
                        pass
                break
        
        for key in ['timeUs', 'averageUs']:
            if key in merged_data:
                try:
                    avg_time = float(merged_data[key]) / 1000.0
                except (ValueError, TypeError):
                    pass
                break
        
        min_time = float(merged_data.get('minMs', merged_data.get('min', avg_time * 0.9)) or avg_time * 0.9)
        max_time = float(merged_data.get('maxMs', merged_data.get('max', avg_time * 1.1)) or avg_time * 1.1)
        
        # 获取精度信息
        precision = 'UNKNOWN'
        
        for key in ['precision', 'Precision', 'format', 'Format', 'dataType', 'dtype']:
            if key in merged_data and merged_data[key]:
                precision = str(merged_data[key]).upper()
                break
        
        if precision == 'UNKNOWN' and 'Outputs' in merged_data:
            outputs = merged_data['Outputs']
            if isinstance(outputs, list) and len(outputs) > 0:
                out = outputs[0]
                if isinstance(out, dict):
                    fmt = out.get('Format/Datatype', '')
                    if fmt:
                        precision = _extract_precision_from_format(fmt)
        
        if precision == 'UNKNOWN' and 'Inputs' in merged_data:
            inputs = merged_data['Inputs']
            if isinstance(inputs, list) and len(inputs) > 0:
                inp = inputs[0]
                if isinstance(inp, dict):
                    fmt = inp.get('Format/Datatype', '')
                    if fmt:
                        precision = _extract_precision_from_format(fmt)
        
        precision = _normalize_precision(precision)
        
        # 获取形状信息
        input_shapes, output_shapes = _extract_shapes(merged_data, name)
        
        # 提取元数据
        metadata = _extract_metadata(merged_data)
        
        return LayerProfile(
            name=name,
            layer_type=layer_type or 'unknown',
            precision=precision,
            avg_time_ms=float(avg_time) if avg_time else 0.0,
            min_time_ms=float(min_time) if min_time else 0.0,
            max_time_ms=float(max_time) if max_time else 0.0,
            percentage=0.0,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            metadata=metadata
        )
    except Exception as e:
        print(f"解析层信息失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def _extract_layers_from_data(data) -> List[dict]:
    """从各种格式的数据中提取层列表"""
    layers = []
    
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                if 'layers' in item or 'Layers' in item:
                    nested = item.get('layers', item.get('Layers', []))
                    layers.extend(nested)
                elif any(key in item for key in ['name', 'Name', 'layerName', 'LayerName', 'averageMs', 'timeMs']):
                    layers.append(item)
                    
    elif isinstance(data, dict):
        for key in ['layers', 'Layers', 'profile', 'Profile']:
            if key in data and isinstance(data[key], list):
                return _extract_layers_from_data(data[key])
        
        for name, info in data.items():
            if isinstance(info, dict):
                info['name'] = name
                layers.append(info)
    
    return layers


def parse_trtexec_json(json_path: str, model_name: str = "", 
                       layer_info_path: str = None) -> Optional[ModelProfile]:
    """解析trtexec输出的JSON文件"""
    if not os.path.exists(json_path):
        print(f"JSON文件不存在: {json_path}")
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 尝试自动查找layerInfo文件
    if layer_info_path is None:
        for suffix in ['_layers.json', '_layerinfo.json', '_layer_info.json']:
            auto_path = json_path.replace('.json', suffix)
            if auto_path != json_path and os.path.exists(auto_path):
                layer_info_path = auto_path
                print(f"[INFO] 自动找到layerInfo文件: {layer_info_path}")
                break
    
    # 加载layerInfo
    layer_info_map = {}
    if layer_info_path and os.path.exists(layer_info_path):
        try:
            with open(layer_info_path, 'r') as f:
                layer_info_data = json.load(f)
            
            layer_info_map = _build_layer_info_map(layer_info_data)
            print(f"[INFO] 加载了 {len(layer_info_map)} 个层的详细信息")
            
        except Exception as e:
            print(f"[WARNING] 加载layerInfo文件失败: {e}")
    
    layers = []
    total_time = 0.0
    
    # 解析profile数据
    profile_layers = _extract_layers_from_data(data)
    
    for layer_data in profile_layers:
        layer = _parse_layer_info(layer_data, layer_info_map)
        if layer:
            layers.append(layer)
            total_time += layer.avg_time_ms
    
    print(f"[DEBUG] 解析到 {len(layers)} 层, 总时间: {total_time:.3f} ms")
    
    # 计算每层占比
    if total_time > 0:
        for layer in layers:
            layer.percentage = (layer.avg_time_ms / total_time) * 100
    
    # 构建ModelProfile
    profile = ModelProfile(
        model_name=model_name or Path(json_path).stem,
        engine_path=json_path.replace('.json', '.engine'),
        total_time_ms=total_time,
        throughput=1000.0 / total_time if total_time > 0 else 0,
        latency_mean_ms=total_time,
        latency_median_ms=total_time,
        latency_99_ms=total_time * 1.1,
        gpu_compute_time_ms=total_time,
        host_walltime_ms=total_time * 1.05,
        layers=layers
    )
    
    return profile


def format_flops(flops: float) -> str:
    """格式化FLOPs显示"""
    if flops >= 1e15:
        return f"{flops/1e15:.2f} PFLOPS"
    elif flops >= 1e12:
        return f"{flops/1e12:.2f} TFLOPS"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f} GFLOPS"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f} MFLOPS"
    elif flops >= 1e3:
        return f"{flops/1e3:.2f} KFLOPS"
    else:
        return f"{flops:.2f} FLOPS"


def format_number(num: float) -> str:
    """格式化大数字显示"""
    if num >= 1e12:
        return f"{num/1e12:.2f}T"
    elif num >= 1e9:
        return f"{num/1e9:.2f}G"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"


def run_trtexec(engine_path: str, output_json: str, 
                warmup: int = 100, iterations: int = 1000,
                batch_size: int = 1) -> bool:
    """运行trtexec进行性能分析"""
    cmd = [
        "trtexec",
        f"--loadEngine={engine_path}",
        f"--exportProfile={output_json}",
        f"--warmUp={warmup}",
        f"--iterations={iterations}",
        f"--batch={batch_size}",
        "--verbose",
        "--dumpProfile",
        "--separateProfileRun",
        "--exportLayerInfo=" + output_json.replace('.json', '_layers.json')
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"trtexec执行失败: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("trtexec执行超时")
        return False
    except FileNotFoundError:
        print("未找到trtexec命令，请确保TensorRT已正确安装")
        return False