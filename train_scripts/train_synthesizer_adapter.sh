export CUDA_VISIBLE_DEVICES=2

PYTHONPATH=../src python ../src/util/train/train_transformer_with_synthesizer_adapter.py \
    "../config/synthesizer_adapter/bible.yaml"
