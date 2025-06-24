#!/bin/bash


num_samples=6400
num_denoising_steps=81
batch_size=64
min_p=0
compile_torch=0

checkpoints=(
    "2025-06-18/12-51-55/checkpoints/latest/ gidd 0"
    "2025-06-13/15-04-32/checkpoints/latest/ gidd 0.2"
    "2025-06-11/18-28-57/checkpoints/latest/ mdlm -"
    "2025-06-01/17-31-45/checkpoints/latest/ gidd 0"
)
datasets=(
    "easy"
    "hard"
)
score_positions=(
    "MDM_max"
    "MDM_margin"
)
select_positions=(
    "top_k_gumbel"
)
update_tokens=(
    "MDM_max"
    "MDM_categorical"
)
ks=(
    "1"
    "2"
)
gumbel_noise_coefficients=(
    "0"
    "0.1"
    "0.5"
)

output_dir="$( dirname "${BASH_SOURCE[0]}" )/../../outputs/evaluate_all"
mkdir -p "$output_dir"
output_dir="$( cd $output_dir && pwd )"
combinations_file="$output_dir/combinations.txt"
> $combinations_file

for dataset in "${datasets[@]}"; do
    suffix="dataset=${dataset} num_samples=$num_samples num_denoising_steps=$num_denoising_steps batch_size=$batch_size min_p=$min_p compile_torch=$compile_torch"
    for entry in "${checkpoints[@]}"; do
        ckpt=$(echo $entry | awk '{print $1}')
        proc=$(echo $entry | awk '{print $2}')
        noise=$(echo $entry | awk '{print $3}')

        if [ "$proc" = "mdlm" ]; then
            # mdlm_vanilla
            for ut in "${update_tokens[@]}"; do
                echo "$ckpt,mdlm_vanilla,update_token=$ut $suffix" >> $combinations_file
            done
            
            # mdlm_adaptive_score_select_update
            for sp in "${score_positions[@]}"; do
                for sel in "${select_positions[@]}"; do
                    if [ "$sel" = "top_k_gumbel" ]; then
                        for k in "${ks[@]}"; do
                            for gn in "${gumbel_noise_coefficients[@]}"; do
                                for ut in "${update_tokens[@]}"; do
                                    echo "$ckpt,mdlm_adaptive_score_select_update,score_position=$sp select_position=$sel update_token=$ut k=$k gumbel_noise_coefficient=$gn $suffix" >> $combinations_file
                                done
                            done
                        done
                    else
                        for ut in "${update_tokens[@]}"; do
                            echo "$ckpt,mdlm_adaptive_score_select_update,score_position=$sp select_position=$sel update_token=$ut $suffix" >> $combinations_file
                        done
                    fi
                done
            done

        elif [ "$proc" = "gidd" ]; then
            # gidd_vanilla_original
            echo "$ckpt,gidd_vanilla_original,$suffix" >> $combinations_file

            # gidd_vanilla_split
            for ut in "${update_tokens[@]}"; do
                echo "$ckpt,gidd_vanilla_split,update_token=$ut $suffix" >> $combinations_file
            done
            
            # gidd_adaptive_score_select_update
            for sp in "${score_positions[@]}"; do
                for sel in "${select_positions[@]}"; do
                    if [ "$sel" = "top_k_gumbel" ]; then
                        for k in "${ks[@]}"; do
                            for gn in "${gumbel_noise_coefficients[@]}"; do
                                for ut in "${update_tokens[@]}"; do
                                    echo "$ckpt,gidd_adaptive_score_select_update,score_position=$sp select_position=$sel update_token=$ut k=$k gumbel_noise_coefficient=$gn $suffix" >> $combinations_file
                                done
                            done
                        done
                    else
                        for ut in "${update_tokens[@]}"; do
                            echo "$ckpt,gidd_adaptive_score_select_update,score_position=$sp select_position=$sel update_token=$ut $suffix" >> $combinations_file
                        done
                    fi
                done
            done

            # gidd_adaptive_change_vs_unmask
            if (( $(echo "$noise > 0" | bc -l) )); then
                for sp in "${score_positions[@]}"; do
                    for sel in "${select_positions[@]}"; do
                        if [ "$sel" = "top_k_gumbel" ]; then
                            for k in "${ks[@]}"; do
                                for gn in "${gumbel_noise_coefficients[@]}"; do
                                    for ut in "${update_tokens[@]}"; do
                                        echo "$ckpt,gidd_adaptive_change_vs_unmask,score_position=$sp select_position=$sel update_token=$ut k=$k gumbel_noise_coefficient=$gn $suffix" >> $combinations_file
                                    done
                                done
                            done
                        else
                            for ut in "${update_tokens[@]}"; do
                                echo "$ckpt,gidd_adaptive_change_vs_unmask,score_position=$sp select_position=$sel update_token=$ut $suffix" >> $combinations_file
                            done
                        fi
                    done
                done
            fi
        fi
    done
done
# cat $combinations_file

csv_file="$output_dir/results.csv"
echo "accuracy,checkpoint,strategy,params" > $csv_file

evaluate_one_py="$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )/evaluate_one.py"
export evaluate_one_py
export csv_file
cat $combinations_file | parallel --colsep ',' -j 3 '
    GPU=$(( ({#} - 1) % 3 + 1 ))

    CKPT={1}
    STRATEGY={2}
    PARAMS_ORIGINAL={3}
    PARAMS=$(echo "$PARAMS_ORIGINAL" | sed "s/\([^ =]*=[^ ]*\)/--\1/g")

    CMD="python $evaluate_one_py --checkpoint $CKPT --strategy $STRATEGY $PARAMS"
    CMD="$CMD --device=$GPU"
    # echo "Running on GPU $GPU: $CMD"
    echo "Running line $(({#}))"
    ACCURACY=$(CUDA_VISIBLE_DEVICES=$GPU $CMD | grep -oP "\K[0-9.]+")
    echo "$ACCURACY,$CKPT,$STRATEGY,\"$PARAMS_ORIGINAL\"" >> $csv_file
'