#!/usr/bin/env python3
"""
极简ONNX检查脚本
"""

import onnx
import sys

def simple_check(model_path):
    """最基本的ONNX检查"""
    try:
        # 加载模型
        model = onnx.load(model_path)
        print(f"✅ 模型加载成功")
        
        # 输入输出
        print(f"\n📥 输入:")
        for inp in model.graph.input:
            print(f"  {inp.name}")
        
        print(f"\n📤 输出:")
        for out in model.graph.output:
            print(f"  {out.name}")
        
        # 层统计
        ops = {}
        for node in model.graph.node:
            ops[node.op_type] = ops.get(node.op_type, 0) + 1
        
        print(f"\n🏗️ 层统计 ({len(model.graph.node)} 层):")
        for op, cnt in sorted(ops.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {op}: {cnt}")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python simple_check.py model.onnx")
        sys.exit(1)
    
    simple_check(sys.argv[1])