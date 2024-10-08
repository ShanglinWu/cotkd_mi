#!/bin/bash

for deduction in "OrIntro" "OrElim" "ProofByContra" "Composed" "AndIntro" "AndElim" 
do
    echo "doing $deduction's generation"
    for i in 1 2 3 4 5 6 7
    do
        trailnum=$(echo "300/$i" | bc)
        echo "doing $i's generation"
        python run_experiment.py --model-name json --model-size text-ada-001 --min-hops $i --max-hops $i --num-trials $trailnum --deduction-rule $deduction --proofs-only --generate_trio
    done
done

echo "All generation processes have been started in the background."