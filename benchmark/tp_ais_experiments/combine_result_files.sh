#!/bin/bash

echo "dims output_samples KLD JSD EVMSE T MEM NESS method target_d accept_rate proposal_samples proposal_evals target_evals" > results.txt

files=`ls $1/def_methods*D.yaml.dat`
for file in $files; do
  echo "Combining file $file"
  sed '1d' $file >> results.txt
done
