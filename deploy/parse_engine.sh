python deploy/parse_engine.py \
--loadEngine ./deploy/NVDeploy/tools/sensitivity_analysis/model/onnx/fcn-resnet50-12_int8.engine \
--exportLayerInfo ./deploy/NVDeploy/tools/sensitivity_analysis/model/onnx/fcn-resnet50-12_int8.json \
--shapes input:1x3x520x520