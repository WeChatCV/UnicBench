MODEL_NAME=qwen-image-edit
MODEL_PATH=/mnt/shenzhen2cephfs/mm-base-vision/kotisye/pretrain/Qwen-Image-Edit
UNICBENCH_DIR=/mnt/shenzhen2cephfs/mm-base-vision/kotisye/data/oteam_edit_500w/benchmark/UnicBench
OUTPUT_DIR=/mnt/shenzhen2cephfs/mm-base-vision/kotisye/result/unicbench

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node 8 \
  -m inference.gen_samples_qwen \
  --model_path ${MODEL_PATH} \
  --unicbench_dir ${UNICBENCH_DIR} \
  --unicbench_path data/test_data.jsonl \
  --output_dir ${OUTPUT_DIR} \
  --model_name ${MODEL_NAME} \
  --seed 42 \
  --languages en