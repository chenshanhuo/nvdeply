#!/usr/bin/env python3
"""
Mark all intermediate layers as outputs in ONNX model for layer-wise analysis.
"""

import onnx
import argparse
from onnx import helper, TensorProto
import os


def mark_selected_ops_as_outputs(input_model_path, output_model_path, op_types):
    """Mark outputs of specified op types as model outputs."""
    model = onnx.load(input_model_path)
    selected_tensors = set()

    # Collect outputs of nodes whose op type is in op_types
    for node in model.graph.node:
        if node.op_type in op_types:
            for output in node.output:
                if output:
                    selected_tensors.add(output)

    # Remove existing outputs and inputs from selected tensors
    existing_outputs = {output.name for output in model.graph.output}
    existing_inputs = {input.name for input in model.graph.input}
    selected_tensors = selected_tensors - existing_outputs - existing_inputs

    print(f"Found {len(selected_tensors)} tensors from ops {op_types} to mark as outputs")

    # Create new outputs for selected tensors
    new_outputs = []
    for tensor_name in sorted(selected_tensors):
        tensor_type = TensorProto.FLOAT  # Default to float
        tensor_shape = []
        for value_info in model.graph.value_info:
            if value_info.name == tensor_name:
                tensor_type = value_info.type.tensor_type.elem_type
                for dim in value_info.type.tensor_type.shape.dim:
                    if dim.dim_value:
                        tensor_shape.append(dim.dim_value)
                    else:
                        tensor_shape.append(None)
                break
        output_tensor = helper.make_tensor_value_info(
            tensor_name,
            tensor_type,
            tensor_shape
        )
        new_outputs.append(output_tensor)

    model.graph.output.extend(new_outputs)
    onnx.save(model, output_model_path)
    print(f"Saved modified model with {len(new_outputs)} additional outputs to: {output_model_path}")
    return len(new_outputs)


def main():
    parser = argparse.ArgumentParser(
        description='Mark intermediate layers as outputs with readable names'
    )
    parser.add_argument('--input', default="onnx/fcn-resnet50-12.onnx", help='Input ONNX model path')
    parser.add_argument('--output', default="onnx/fcn-resnet50-12-marked_container.onnx", 
                        help='Output ONNX model path')
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

    num_outputs = mark_selected_ops_as_outputs(args.input, args.output, op_types)

    print(f"\nNext steps:")
    print(f"1. Convert the modified ONNX to TensorRT engines:")
    print(f"   trtexec --onnx={args.output} --saveEngine=engine_with_intermediate.engine --dumpProfile --profilingVerbosity=detailed")
    print(f"   trtexec --onnx={args.output} --saveEngine=engine2_with_intermediate.engine --dumpProfile --profilingVerbosity=detailed")
    print(f"2. Run sensitivity analysis:")
    print(f"   python layer_precision_sensitivity_analysis.py --engine engine_with_intermediate.engine --engine2 engine2_with_intermediate.engine")


if __name__ == "__main__":
    ###简易版本，添加的输出名称可能为数字，不易读取
    main()