#!/bin/bash

files=`ls $1/results*D.txt`
for file in $files; do
  echo "Combining file $file"
  sed '1d' $file >> results.txt
done
