export CUDA_VISIBLE_DEVICES=0

PYTHONPATH=../src python ../src/util/test/test_multi_adapter.py \
    "../config/parallel_adapter/iwslt.yaml"
