#!/bin/bash
# Usage: ./run_evaluate_strategies_from_csv.sh <number_of_iterations>
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_file> <output_file> <number_of_iterations>"
    exit 1
fi
INPUT_FILE=$1
OUTPUT_FILE=$2
NUM_ITERATIONS=$3

echo "Input file: $INPUT_FILE"
echo "Output files: $OUTPUT_FILE'_iteration_i' for i in 1 to $NUM_ITERATIONS"
for i in $(seq 1 $NUM_ITERATIONS); do
    OUTPUT_FILE_I="${OUTPUT_FILE%.csv}_iteration_${i}.csv"
    
    echo "Running evaluation for iteration $i..."
    python ./gidd/eval/evaluate_sampling_strategies_from_csv.py --config-name evaluate_sampling_strategies_from_csv input_path=$INPUT_FILE output_path=$OUTPUT_FILE_I shuffle_ds=True
done
echo "All evaluations completed."