export CUDA_VISIBLE_DEVICES=0

PYTHONPATH=../src python ../src/util/test/test_classifier_adapter.py \
    "../config/classifier_adapter/books.yaml"
