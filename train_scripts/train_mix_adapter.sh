export CUDA_VISIBLE_DEVICES=3

PYTHONPATH=../src python ../src/util/train/train_transformer_with_mix_adapter.py \
    "../config/mix_adapter/books.yaml"
