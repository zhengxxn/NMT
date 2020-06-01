 #!/bin/bash

 mosesdecoder="/home/user_data/zhengzx/mt/mosesdecoder"
 
 # (zzx) reference should be previously tokenized by "$mosesdecoder/scripts/tokenizer/tokenizer.perl" 
 gold_targets="/home/zhengx/data/wmt2014-ende-share/newstest2014.de"

 decodes_file="/home/zhengx/output/wmt2014-ende/newstest2014.output"

 # Replace unicode.
 perl $mosesdecoder/scripts/tokenizer/replace-unicode-punctuation.perl -l de  < $decodes_file > $decodes_file.n
 perl $mosesdecoder/scripts/tokenizer/replace-unicode-punctuation.perl -l de  < $gold_targets > $gold_targets.n
 # Tokenize.
 perl $mosesdecoder/scripts/tokenizer/tokenizer.perl -l de < $decodes_file.n > $decodes_file.tok
 perl $mosesdecoder/scripts/tokenizer/tokenizer.perl -l de < $gold_targets.n > $gold_targets.tok
 # Put compounds in ATAT format (comparable to papers like GNMT, ConvS2S).
 # See https://nlp.stanford.edu/projects/nmt/ :
 # 'Also, for historical reasons, we split compound words, e.g.,
 #    "rich-text format" --> rich ##AT##-##AT## text format."'
 perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $gold_targets.tok > $gold_targets.tok.atat
 perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $decodes_file.tok > $decodes_file.tok.atat

 # Get BLEU.
 perl $mosesdecoder/scripts/generic/multi-bleu.perl $gold_targets.tok.atat < $decodes_file.tok.atat

