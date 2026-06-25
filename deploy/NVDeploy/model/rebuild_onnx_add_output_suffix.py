#!/usr/bin/env python3
"""
Mark intermediate layers as outputs WITHOUT changing internal node names.
只修改输出名称，不修改原始层名称。
"""

import onnx
import argparse
from onnx import helper, TensorProto, shape_inference
import os
from collections import defaultdict


def mark_intermediate_outputs(input_model_path, output_model_path, op_types):
    """标记中间层为输出，但保持内部节点名称不变"""
    print(f"Loading model: {input_model_path}")
    model = onnx.load(input_model_path)
    
    # 运行形状推断
    try:
        print("Running shape inference...")
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Warning: Shape inference failed: {e}")
    
    # 统计各种操作类型的数量
    op_type_counts = defaultdict(int)
    
    # 遍历所有节点，统计要标记的操作类型
    for node in model.graph.node:
        if node.op_type in op_types:
            op_type_counts[node.op_type] += 1
    
    print(f"Found operation counts: {dict(op_type_counts)}")
    
    # 收集要标记的输出
    selected_tensors = []
    op_type_current_counts = defaultdict(int)
    
    for node in model.graph.node:
        if node.op_type in op_types and node.output:
            op_type_current_counts[node.op_type] += 1
            
            # 获取节点的原始输出名称
            original_output_name = node.output[0]
            
            # 创建易读的输出名称
            if node.name and node.name.strip():
                # 使用节点名称 + _output
                new_output_name = f"{node.name}_output"
            else:
                # 使用操作类型 + 序号 + _output
                count = op_type_current_counts[node.op_type]
                new_output_name = f"{node.op_type.lower()}{count}_output"
            
            # 确保名称唯一
            existing_names = {out.name for out in model.graph.output}
            temp_name = new_output_name
            suffix = 1
            while temp_name in existing_names:
                temp_name = f"{new_output_name}_{suffix}"
                suffix += 1
            new_output_name = temp_name
            
            selected_tensors.append({
                'original_name': original_output_name,
                'new_name': new_output_name,
                'node': node
            })
    
    print(f"\nWill mark {len(selected_tensors)} tensors as outputs")
    
    if len(selected_tensors) == 0:
        print("Warning: No tensors found to mark as outputs!")
        onnx.save(model, output_model_path)
        return 0
    
    # 添加重命名节点和新输出 
    new_outputs = []
    
    for tensor_info in selected_tensors:
        original_name = tensor_info['original_name']
        new_name = tensor_info['new_name']
        node = tensor_info['node']
        
        # 查找张量信息
        tensor_type = TensorProto.FLOAT
        tensor_shape = []
        
        # 在value_info中查找
        found = False
        for value_info in model.graph.value_info:
            if value_info.name == original_name:
                tensor_type = value_info.type.tensor_type.elem_type
                for dim in value_info.type.tensor_type.shape.dim:
                    if dim.HasField('dim_value'):
                        tensor_shape.append(dim.dim_value)
                    elif dim.HasField('dim_param'):
                        tensor_shape.append(dim.dim_param)
                    else:
                        tensor_shape.append(None)
                found = True
                break
        
        if not found:
            print(f"  Warning: Could not find shape info for {original_name}")
        
        try:
            # 添加Identity节点来重命名输出 避免添加的output节点名称为随机名称
            identity_node = helper.make_node(
                'Identity',
                inputs=[original_name],
                outputs=[new_name],
                name=f"rename_{new_name}"
            )
            
            # 在原始节点之后插入Identity节点
            # 查找节点索引
            node_index = -1
            for i, n in enumerate(model.graph.node):
                if n == node:
                    node_index = i
                    break
            
            if node_index >= 0:
                # 在节点之后插入Identity节点
                model.graph.node.insert(node_index + 1, identity_node)
            else:
                # 如果找不到，添加到末尾
                model.graph.node.append(identity_node)
            
            # 创建新的输出
            output_tensor = helper.make_tensor_value_info(
                new_name,
                tensor_type,
                tensor_shape
            )
            
            new_outputs.append(output_tensor)
            
            print(f"  ✓ {node.op_type} -> {new_name} (from {original_name})")
            
        except Exception as e:
            print(f"  ✗ Error creating output for {original_name}: {e}")
    
    # 添加新的输出到模型
    if new_outputs:
        model.graph.output.extend(new_outputs)
        
        # 验证模型
        print("\nValidating model...")
        try:
            onnx.checker.check_model(model)
            print("✓ Model validation passed")
        except Exception as e:
            print(f"⚠ Model validation warning: {e}")
        
        # 保存模型
        onnx.save(model, output_model_path)
        print(f"\n✓ Saved model to: {output_model_path}")
        print(f"  Added {len(new_outputs)} new outputs")
        print(f"  Total outputs: {len(model.graph.output)}")
        
        # 显示一些示例输出名称
        print(f"\nSample output names (first 10):")
        for i, out in enumerate(model.graph.output[-10:]):
            print(f"  {i+1}. {out.name}")
    else:
        print("✗ No new outputs were created")
        onnx.save(model, output_model_path)
    
    return len(new_outputs)


def main():
    parser = argparse.ArgumentParser(
        description='Mark intermediate layers as outputs with readable names'
    )
    parser.add_argument('--input', default="onnx/fcn-resnet50-12.onnx", help='Input ONNX model path')
    parser.add_argument('--output', default="onnx/fcn-resnet50-12-marked_container.onnx", help='Output ONNX model path')
    parser.add_argument('--ops', required=True, 
                       help='Comma-separated list of op types to mark, e.g. Conv,Relu,Add')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input model '{args.input}' not found!")
        return
    
    op_types = [op.strip() for op in args.ops.split(',') if op.strip()]
    if not op_types:
        print("Error: No op types specified with --ops")
        return
    
    print(f"Marking outputs for operation types: {op_types}")
    
    num_outputs = mark_intermediate_outputs(args.input, args.output, op_types)
    
    #后续操作提示
    if num_outputs > 0:
        print(f"\nNext steps:")
        print(f"1. Convert to TensorRT engines:")
        print(f"   trtexec --onnx={args.output} --saveEngine=fp16.engine --fp16")
        print(f"   trtexec --onnx={args.output} --saveEngine=int8.engine --int8")
        print(f"2. Run sensitivity analysis:")
        print(f"   python sensitivity_analysis.py \\")
        print(f"     --engine1 fp16.engine \\")
        print(f"     --engine2 int8.engine")


if __name__ == "__main__":
    main()