#!/bin/bash

corpus_path="/home/gh/autosubs/data/kenlm_knnw.txt"
out_path="/home/gh/autosubs/data/kenlm_knnw.arpa"
order=3

/home/gh/kenlm/build/bin/lmplz -o ${order} -S 40% -T /tmp <${corpus_path}  >${out_path}
