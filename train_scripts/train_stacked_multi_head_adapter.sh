export CUDA_VISIBLE_DEVICES=2

PYTHONPATH=../src python ../src/util/train/train_transformer_with_stacked_multi_head_adapter.py \
    "../config/stacked_multi_head_adapter/laws.yaml"
