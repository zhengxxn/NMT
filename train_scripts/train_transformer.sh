export CUDA_VISIBLE_DEVICES=5

PYTHONPATH=../src python ../src/util/train/train_transformer.py \
    "../config/translation-transformer/laws-ende.yaml"
