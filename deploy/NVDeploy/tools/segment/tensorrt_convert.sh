# 动态形状
python deploy/NVDeploy/tools/segment/tensorrt_converter.py \
     --onnx deploy/NVDeploy/model/onnx/fcn-resnet50-12-marked_contain.onnx \
     --dynamic \
    --opt-shape 1,3,520,520 \
    --max-shape 4,3,520,520 \
    --min-shape 1,3,520,520 \
    --int8 \
    --calib-data-dir data/fcn-resnet50-12/calib \
    --mixed-precision \
    --fp16-layers "Conv_6,Relu_5"