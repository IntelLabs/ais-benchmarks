#!/usr/bin/env bash

#methods="BSPT random MCMC-MH grid nested multi-nested reject"
distributions="gmm normal rosenbrock ripple"
dimensions="1 2 3 4 5 6 7 8 9 10"

m=$1
for d in $distributions
do
    for dim in $dimensions
    do
        echo "python3 sampling_examples.py $m $d $dim"
        python3 sampling_examples.py $m $d $dim &
    done
done
