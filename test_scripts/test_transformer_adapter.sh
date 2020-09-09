export CUDA_VISIBLE_DEVICES=3

PYTHONPATH=../src python ../src/util/test/test_transformer_adapter.py \
    "../config/kd_adapter/php.yaml"
