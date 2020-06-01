export CUDA_VISIBLE_DEVICES=0

PYTHONPATH=../src python ../src/util/test/test_split_position_transformer.py \
    "../config/split_position/medical-ende.yaml"
