#!/bin/bash


num_samples=6400
num_denoising_steps=81
batch_size=64
min_p=0
compile_torch=0

seeds=(
    "1"
    "2"
    "3"
)
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
score_positions_change=(
    "change_max"
    "change_margin"
)
select_positions=(
    "top_k_gumbel"
)
select_positions_change=(
    "all"
)
update_tokens=(
    "MDM_max"
    "MDM_categorical"
)
update_tokens_change=(
    "change_max"
    "change_categorical"
)
ks=(
    "1"
)
gumbel_noise_coefficients=(
    "0"
)

output_dir="$( dirname "${BASH_SOURCE[0]}" )/../../outputs/evaluate_all"
mkdir -p "$output_dir"
output_dir="$( cd $output_dir && pwd )"
combinations_file="$output_dir/combinations.txt"
> $combinations_file
for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
        suffix="dataset=${dataset} num_samples=$num_samples num_denoising_steps=$num_denoising_steps batch_size=$batch_size min_p=$min_p compile_torch=$compile_torch seed=$seed"
        for entry in "${checkpoints[@]}"; do
            ckpt=$(echo $entry | awk '{print $1}')
            proc=$(echo $entry | awk '{print $2}')
            noise=$(echo $entry | awk '{print $3}')

            if [ "$proc" = "mdlm" ]; then
                # mdlm_vanilla # should be equivalent to gidd_vanilla_split for p_u=0
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
                # gidd_vanilla_original # should be equivalent to gidd_vanilla_independent_change if the token is sampled categorially, but not if max is used (except that here, unmasked tokens can cange back to masks)
                # Every position is updated in every step as follows:
                # For each position, compute the probability p_i that the new token is token i, for all i in the vocabulary. And then sample categorically from this distribution.
                # Note: the mask token is also in the vocabulary, thus unmasked tokens can change back to masks. Also, tokens can very well stay the same in a step, according to the probability for that.
                # TODO: check whether unmasked tokens can actually change back to masks
                echo "$ckpt,gidd_vanilla_original,$suffix" >> $combinations_file

                # gidd_vanilla_split # unmasking only, should be equivalent to gidd_vanilla_independent_change if p_u=0
                # For each position independently decide to unmask the token with probability p_unmask,
                # p_unmask depends only on the time step and is thus the same for all positions.
                # Parameters:
                # - The strategy to select the new token once a position decides to unmask the current token.
                # Note: unmasked tokens never change, thus the stopping criterion is that all tokens are unmasked.
                for ut in "${update_tokens[@]}"; do
                    echo "$ckpt,gidd_vanilla_split,update_token=$ut $suffix" >> $combinations_file
                done

                # gidd_vanilla_independent_change # if p_u=0, should be equivalent to gidd_vanilla_split
                # For each position independently decide to change the token with probability p_change,
                # p_change depends on the model's prediction and the current token value and can thus vary from position to position.
                # Parameters:
                # - The strategy to select the new token once a position decides to change the current token.
                # - The stopping criterion (e.g. no masks left)
                # Note: unmasked tokens can change into other tokens, but never back to masks
                for ut in "${update_tokens_change[@]}"; do
                    echo "$ckpt,gidd_vanilla_independent_change,update_token=$ut $suffix" >> $combinations_file
                done
                
                # gidd_adaptive_score_select_update
                # For each position compute a score based on the model's prediction.
                # Based on the scores of the positions, select a subset of positions to update.
                # Update the selected positions according to your strategy.
                # Parameters:
                # - The function to compute the score for each position.
                # - The strategy to select the positions to update based on the scores.
                # - The strategy to select the new token once a position is selected.
                # Note: current implementation does not allow tokens to change once unmasked (designed for p_u=0, should be equivalent to mdlm_adaptive_score_select_update)
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
                # For each position compute the probability to change the token based on the model's prediction.
                # Also compute the probability to unmask at the current time step. This does not depend on the position.
                # Check if there is at least one position where the probability to change is greater than the probability to unmask.
                # If so, change a subset of those positions according to your "change_strategy". The subset is selected according to the "select_position_change" strategy.
                # If not, choose a subset of currently masked positions and unmask them according to your "unmask_strategy". The subset is selected according to the "select_position_unmask" strategy.
                # Parameters:
                # - "change_strategy", composed of a strategy to select subset of the eligible positions to change, and strategy to select the new token once a position is selected.
                # - "unmask_strategy", potentially composed of a score function for each masked position, strategy to select the positions to unmask, and strategy to select the new token once a position is selected.
                # Note: the "change_strategy" should not allow unmasked tokens to turn back into masks.
                if (( $(echo "$noise > 0" | bc -l) )); then
                    for sp in "${score_positions[@]}"; do
                        for sel_change in "${select_positions_change[@]}"; do
                            for sel_unmask in "${select_positions[@]}"; do
                                if [ "$sel_unmask" = "top_k_gumbel" ]; then
                                    for k in "${ks[@]}"; do
                                        for gn in "${gumbel_noise_coefficients[@]}"; do
                                            for ut_change in "${update_tokens_change[@]}"; do
                                                for ut_unmask in "${update_tokens[@]}"; do
                                                    echo "$ckpt,gidd_adaptive_change_vs_unmask,score_position=$sp select_position_change=$sel_change select_position_unmask=$sel_unmask update_token_change=$ut_change update_token_unmask=$ut_unmask k=$k gumbel_noise_coefficient=$gn $suffix" >> $combinations_file
                                                done
                                            done
                                        done
                                    done
                                else
                                    for ut_change in "${update_tokens_change[@]}"; do
                                        for ut_unmask in "${update_tokens[@]}"; do
                                            echo "$ckpt,gidd_adaptive_change_vs_unmask,score_position=$sp select_position_change=$sel_change select_position_unmask=$sel_unmask update_token_change=$ut_change update_token_unmask=$ut_unmask $suffix" >> $combinations_file
                                        done
                                    done
                                fi
                            done
                        done
                    done
                    
                fi

                # gidd_change_based_on_model_confidence_to_change
                # For each position, compute the score as confidence of the model on any token that is different from the current token.
                # Based on the scores of the positions, select a subset of positions to update.
                # Update the selected positions according to your strategy.
                # Parameters:
                # - The function to compute the score for each position.
                # - The strategy to select the positions to update based on the scores.
                # - The strategy to select the new token once a position is selected.
                if (( $(echo "$noise > 0" | bc -l) )); then
                    for sp in "${score_positions_change[@]}"; do
                        for sel in "${select_positions[@]}"; do
                            if [ "$sel" = "top_k_gumbel" ]; then
                                for k in "${ks[@]}"; do
                                    for gn in "${gumbel_noise_coefficients[@]}"; do
                                        for ut in "${update_tokens_change[@]}"; do
                                            echo "$ckpt,gidd_change_based_on_model_confidence_to_change,score_position=$sp select_position=$sel update_token=$ut k=$k gumbel_noise_coefficient=$gn $suffix" >> $combinations_file
                                        done
                                    done
                                done
                            else
                                for ut in "${update_tokens_change[@]}"; do
                                    echo "$ckpt,gidd_change_based_on_model_confidence_to_change,score_position=$sp select_position=$sel update_token=$ut $suffix" >> $combinations_file
                                done
                            fi
                        done
                    done
                fi
            fi
        done
    done
done
# cat $combinations_file

csv_file="$output_dir/results.csv"
echo "accuracy,correctly_filled_cells,checkpoint,strategy,params" > $csv_file

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
    OUTPUT=$(CUDA_VISIBLE_DEVICES=$GPU $CMD)
    ACCURACY=$(echo "$OUTPUT" | grep -oP "accuracy=\K[0-9.]+")
    CORRECTLY_FILLED_CELLS=$(echo "$OUTPUT" | grep -oP "correctly_filled_cells=\K[0-9.]+")
    echo "$ACCURACY,$CORRECTLY_FILLED_CELLS,$CKPT,$STRATEGY,\"$PARAMS_ORIGINAL\"" >> $csv_file
'