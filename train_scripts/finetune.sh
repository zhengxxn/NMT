export CUDA_VISIBLE_DEVICES=7

PYTHONPATH=../src python ../src/util/train/finetune_subnetwork.py \
    "../config/finetune_subnetwork/laws.yaml"
