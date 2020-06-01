export CUDA_VISIBLE_DEVICES=0

PYTHONPATH=../src python ../src/util/test/test_transformer_adapter.py \
    "../config/transformer-adapter/wmt2014-ende-entire.yaml"
