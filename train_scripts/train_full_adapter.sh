export CUDA_VISIBLE_DEVICES=3

PYTHONPATH=../src python ../src/util/train/train_transformer_with_full_adapter.py \
    "../config/full_adapter/laws.yaml"
