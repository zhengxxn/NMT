export CUDA_VISIBLE_DEVICES=1

PYTHONPATH=../src python ../src/util/train/train_transformer_with_diff_size_stacked_adapter.py \
    "../config/diff_size_adapter/laws.yaml"
