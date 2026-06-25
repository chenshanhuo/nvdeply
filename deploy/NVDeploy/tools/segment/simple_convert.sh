# 1. 基本转换 (FP32)
echo "正在执行基本转换 (FP32)..."
python deploy/NVDeploy/tools/sensitivity_analysis/segment/simple_convert.py \
--onnx deploy/NVDeploy/tools/sensitivity_analysis/model/onnx/fcn-resnet50-12.onnx

# 2. 启用 FP16
echo "正在执行 FP16 转换..."
python deploy/NVDeploy/tools/sensitivity_analysis/segment/simple_convert.py \
--onnx deploy/NVDeploy/tools/sensitivity_analysis/model/onnx/fcn-resnet50-12.onnx \
--fp16

# 3. 指定输出路径
echo "正在执行转换并指定输出路径..."
python deploy/NVDeploy/tools/sensitivity_analysis/segment/simple_convert.py \
--onnx deploy/NVDeploy/tools/sensitivity_analysis/model/onnx/fcn-resnet50-12.onnx \
--output my_engine.engine \
--fp16

# 4. 动态输入形状
echo "正在执行动态输入形状转换..."
python deploy/NVDeploy/tools/sensitivity_analysis/segment/simple_convert.py \
--onnx deploy/NVDeploy/tools/sensitivity_analysis/model/onnx/fcn-resnet50-12.onnx \
--dynamic \
--height 480 \
--width 640 \
--fp16

# # 5. 增加工作空间
# echo "正在执行增加工作空间的转换..."
# python deploy/NVDeploy/tools/sensitivity_analysis/segment/debug/simple_convert.py --onnx fcn-resnet50-12.onnx --workspace 4096 --fp16