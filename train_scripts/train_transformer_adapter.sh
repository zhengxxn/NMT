export CUDA_VISIBLE_DEVICES=3

PYTHONPATH=../src python ../src/util/train/train_transformer_with_adapter.py \
    "../config/transformer-adapter/medical-ende.yaml"
