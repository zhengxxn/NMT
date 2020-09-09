export CUDA_VISIBLE_DEVICES=3

PYTHONPATH=../src python ../src/util/train/train_with_parallel_adapter.py \
    "../config/parallel_adapter/bible.yaml"
