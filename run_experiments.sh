#!/usr/bin/env bash

trials="2 3 4 5 6 7 8 9 10"

for i in $trials
do
    echo "python3 evaluate_methods.py results_$i.txt"
    python3 evaluate_methods.py results_$i.txt
done
