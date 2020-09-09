export CUDA_VISIBLE_DEVICES=0

PYTHONPATH=../src python ../src/util/test/test_mix_adapter.py \
    "../config/mix_adapter/mix-GNOME.yaml"
