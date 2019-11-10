#!/usr/bin/env bash

trials=`seq 0 $1`

date
for i in $trials
do
    echo "python3 evaluate_methods.py results_$i.txt"
    python3 evaluate_methods.py results_${i}.txt 2>results_${i}_err.txt &
    sleep 0.1
done
wait
date
