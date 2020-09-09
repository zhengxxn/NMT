export CUDA_VISIBLE_DEVICES=6

PYTHONPATH=../src python ../src/util/train/train_transformer.py \
    "../config/translation-transformer/ubuntu-ende.yaml"
