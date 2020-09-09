export CUDA_VISIBLE_DEVICES=3

PYTHONPATH=../src python ../src/util/test/test_transformer.py \
    "../config/translation-transformer/wmt2014-ende-new.yaml"
