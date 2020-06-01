export CUDA_VISIBLE_DEVICES=0

PYTHONPATH=../src python ../src/util/average_models.py \
    "../config/split_position/wmt2014-ende-entire.yaml"
