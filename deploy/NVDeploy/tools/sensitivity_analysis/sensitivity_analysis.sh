

python deploy/NVDeploy/tools/sensitivity_analysis/sensitivity_analysis.py \
--engine1 deploy/NVDeploy/model/onnx/fcn-resnet50-12-marked_contain_fp16.engine \
--engine2 deploy/NVDeploy/model/onnx/fcn-resnet50-12-marked_contain_int8_mixed.engine