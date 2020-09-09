export CUDA_VISIBLE_DEVICES=7

PYTHONPATH=../src python ../src/util/train/train_transformer_with_houlsby_adapter.py \
    "../config/houlsby_adapter/iwslt2015-ende.yaml"
