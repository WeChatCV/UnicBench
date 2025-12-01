MODEL_NAME=flux_kontext
MODEL_PATH=black-forest-labs/FLUX.1-Kontext-dev
UNICBENCH_DIR=/path/to/UnicBench/images
OUTPUT_DIR=/path/to/output

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node 8 \
  -m inference.gen_samples_flux \
  --model_path ${MODEL_PATH} \
  --unicbench_dir ${UNICBENCH_DIR} \
  --unicbench_path data/test_data.jsonl \
  --output_dir ${OUTPUT_DIR} \
  --model_name ${MODEL_NAME} \
  --seed 42 \
  --languages en