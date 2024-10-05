for i in 3 4 5 6 7 8
do
    echo "doing $i's generation"
    python run_experiment.py --model-name json --model-size text-ada-001 --min-hops $i --max-hops $i --num-trials 500 --generate_trio #with distractors
done    