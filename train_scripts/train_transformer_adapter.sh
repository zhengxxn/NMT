export CUDA_VISIBLE_DEVICES=0

PYTHONPATH=../src python ../src/util/train/train_transformer_with_adapter.py \
    "../config/transformer-adapter/iwslt2015-ende.yaml"
