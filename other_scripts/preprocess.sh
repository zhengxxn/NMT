PYTHONPATH=../src python ../src/util/data_preprocess.py \
   "../config/preprocess/wmt14-adapt.yaml" \
   "token->clean->apply_bpe->save_tsv"
