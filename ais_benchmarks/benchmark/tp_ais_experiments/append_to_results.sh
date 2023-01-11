#!/bin/bash

files=`ls $1/def_methods*D.yaml.dat`
for file in $files; do
  echo "Appending file $file"
  sed '1d' $file >> results.txt
done
