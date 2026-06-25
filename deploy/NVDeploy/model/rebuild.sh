###用于将onnx模型中的感兴趣节点进行输出，方便后续量化过程中逐层对比

python "./deploy/NVDeploy/model/rebuild_onnx_add_output_suffix.py" \
--input './deploy/NVDeploy/model/onnx/fcn-resnet50-12.onnx' \
--output 'deploy/NVDeploy/model/onnx/fcn-resnet50-12-marked_contain.onnx' \
--ops Conv 