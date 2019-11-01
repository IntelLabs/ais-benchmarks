#!/usr/bin/env bash

trials=`seq 0 56`

for i in $trials
do
    echo "python3 evaluate_methods.py results_$i.txt"
    python3 evaluate_methods.py results_$i.txt &
    sleep 0.1
done
