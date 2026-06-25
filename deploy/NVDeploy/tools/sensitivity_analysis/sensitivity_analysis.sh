

python deploy/NVDeploy/tools/sensitivity_analysis/sensitivity_analysis.py \
--engine1 deploy/NVDeploy/tools/sensitivity_analysis/model/onnx/fcn-resnet50-12-marked_container_fp16.engine \
--engine2 deploy/NVDeploy/tools/sensitivity_analysis/model/onnx/fcn-resnet50-12-marked_container_int8.engine