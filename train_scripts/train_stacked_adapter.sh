export CUDA_VISIBLE_DEVICES=3

PYTHONPATH=../src python ../src/util/train/train_with_stacked_adapter.py \
    "../config/stacked_adapter/laws.yaml"
