#!/bin/bash

# 设置输出目录
OUTPUT_DIR="output/profile"
mkdir -p ${OUTPUT_DIR}

# #对于fp16模型
# trtexec --loadEngine=deploy/NVDeploy/model/onnx/fcn-resnet50-12_fp16.engine \
#         --exportProfile=${OUTPUT_DIR}/original_profile.json \
#         --exportLayerInfo=${OUTPUT_DIR}/original_layers.json \
#         --profilingVerbosity=detailed \
#         --warmUp=100 \
#         --iterations=1000 \
#         --dumpProfile

# # 对于mixed模型
# trtexec --loadEngine=deploy/NVDeploy/model/onnx/fcn-resnet50-12_int8_mixed.engine \
#         --exportProfile=${OUTPUT_DIR}/mixed_profile.json \
#         --exportLayerInfo=${OUTPUT_DIR}/mixed_layers.json \
#         --profilingVerbosity=detailed \
#         --warmUp=100 \
#         --iterations=1000 \
#         --dumpProfile

# 运行分析对比
python deploy/NVDeploy/tools/model_evalution/main.py compare \
    --baseline output/profile/fp16_profile.json \
    --baseline-layers output/profile/fp16_layers.json \
    --optimized output/profile/mixed_profile.json \
    --optimized-layers output/profile/mixed_layers.json \
    --baseline-name "fp16 Model" \
    --optimized-name "mixed Model" \
    --report \
    --interactive

# # 分析单个模型 Original Model
# python deploy/NVDeploy/tools/model_evalution/main.py analyze \
#     --json output/profile/fp16_profile.json \
#     --layers output/profile/fp16_layers.json \
#     --name "fp16 Model" \
#     --report
