export CUDA_VISIBLE_DEVICES=0

PYTHONPATH=../src python ../src/util/test/test_transformer.py \
    "../config/translation-transformer/medical-ende.yaml"
