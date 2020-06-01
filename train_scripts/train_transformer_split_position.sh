export CUDA_VISIBLE_DEVICES=7

PYTHONPATH=../src python ../src/util/train/train_split_pos_transformer.py \
    "../config/split_position/laws-ende.yaml"
