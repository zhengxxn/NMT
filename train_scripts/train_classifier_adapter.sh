export CUDA_VISIBLE_DEVICES=3

PYTHONPATH=../src python ../src/util/train/train_transformer_with_classifier_adapter.py \
    "../config/classifier_adapter/books.yaml"
