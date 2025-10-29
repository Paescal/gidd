#!/bin/bash
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export VECLIB_MAXIMUM_THREADS=3

num_jobs=8

num_samples=6400
batch_size=64
min_p=0
compile_torch=0


beam_search=(
    false
    true
)
steps_before_pruning=(
    # "1"
    # "2"
    # "4"
    "8"
    # "16"
    # "32"
)
pruning_num_beams=(
    # "1"
    "2"
)
branching_factors=(
    # "2"
    # "3"
    "4"
    # "8"
)
score_times=(
    "t_zero"
    # "inferred"
)
score_methods=(
    "avg"
    # "min"
)
initiate_beam_search_after_progress=(
    # "0.0"
    "0.1"
    # "0.5"
)
seeds=(
    "1"
    # "2"
    # "3"
)
checkpoints=(
    # "2025-06-18/12-51-55/checkpoints/latest/ gidd 0"
    # "2025-06-13/15-04-32/checkpoints/latest/ gidd 0.2"
    # "2025-06-11/18-28-57/checkpoints/latest/ mdlm -"
    # "2025-06-01/17-31-45/checkpoints/latest/ gidd 0"
    # "checkpoints/mdlm/50_epochs mdlm -"
    # "checkpoints/mdlm/100_epochs mdlm -"
    "checkpoints/mdlm/300_epochs mdlm -"
    # "checkpoints/gidd_0/50_epochs gidd 0"
    # "checkpoints/gidd_0/100_epochs gidd 0"
    "checkpoints/gidd_0/300_epochs gidd 0"
    # "checkpoints/gidd_0_2/50_epochs gidd 0.2"
    # "checkpoints/gidd_0_2/100_epochs gidd 0.2"
    "checkpoints/gidd_0_2/300_epochs gidd 0.2"
)
datasets=(
    # "easy"
    "hard"
)
nums_denoising_steps=(
    # "5"
    # "10"
    # "20"
    # "30"
    # "40"
    # "50"
    # "60"
    # "70"
    # "80"
    # "90"
    # "100"
    # "150"
    # "200"
    # "300"
    "81"
    # "128"
)
nums_self_correction_steps=(
    # "0"
    # "4"
    "32"
)
strategies=(
    # "mdlm_vanilla"
    # "mdlm_adaptive_score_select_update"
    # "gidd_emulate_mdlm_vanilla"
    # "gidd_emulate_mdlm_adaptive_score_select_update"
    # "gidd_original"
    # "gidd_independent_positions_decomposed_update_distribution"
    "gidd_selected_positions_decomposed_update_distribution"
    "gidd_change_based_on_model_confidence_to_change"
    "gidd_keep_where_confident"
    "gidd_change_low_confidence_positions"
    # "gidd_flattened"
    "gidd_prob_to_recover_data"
)
score_mask_positions=(
    "MDM_max"
    # "MDM_margin"
)
score_positions_for_change=(
    "change_max"
    # "change_margin"
)
select_positions=(
    "top_k_gumbel"
)
select_positions_to_change=(
    "all"
)
unmask_tokens=(
    "MDM_max"
    # "MDM_categorical"
)
change_tokens=(
    "change_max"
    # "change_categorical"
)
ks=(
    "1"
    # "2"
    # "3"
    # "4"
    # "5"
    # "6"
    # "7"
    # "8"
    # "9"
    # "10"
    # "15"
)
gumbel_noise_coefficients=(
    "0"
)
self_correction_strategies=(
    "none"
    # "original"
    # "oscillation_prevention_fast"
    # "oscillation_prevention_slow"
    # "keep_where_confident"
    # "max"
)
oracles=(
    # "perfect"
    # "model"
    # "recurrence"
    # "model_and_recurrence"
    "model_EMA"
)
position_sampling_strategies=(
    # "independent"
    "top_k"
)
position_metric_strategies=(
    # "p_denoise"
    # "confident_and_p_denoise"
    # "confident_and_noisy"
    "margin_and_noisy"
    # "noisy"
    # "confident"
)
token_sampling_strategies=(
    # "categorical"
    # "change_max"
    "max"
)
uniform_noise_strategies=(
    "none"
    # "noise"
    # "model"
)

output_dir="$( dirname "${BASH_SOURCE[0]}" )/../../outputs/evaluate_all"
mkdir -p "$output_dir"
output_dir="$( cd $output_dir && pwd )"
combinations_file="$output_dir/combinations_debug.txt"
csv_file="$output_dir/results_debug.csv"

# Function to process checkpoints with given suffix
process_checkpoints() {
    local suffix="$1"
    
    for entry in "${checkpoints[@]}"; do
        ckpt=$(echo $entry | awk '{print $1}')
        proc=$(echo $entry | awk '{print $2}')
        noise=$(echo $entry | awk '{print $3}')

        # -------------------- MDLM --------------------
        if [ "$proc" = "mdlm" ]; then
            # mdlm_vanilla # should be equivalent to gidd_emulate_mdlm_vanilla for p_u=0
            if [[ " ${strategies[@]} " =~ " mdlm_vanilla " ]]; then
                for unmask_token in "${unmask_tokens[@]}"; do
                    echo "$ckpt,mdlm_vanilla,unmask_token=$unmask_token $suffix" >> $combinations_file
                done
            fi
            
            # mdlm_adaptive_score_select_update
            if [[ " ${strategies[@]} " =~ " mdlm_adaptive_score_select_update " ]]; then
                for score_mask_position in "${score_mask_positions[@]}"; do
                    for select_position in "${select_positions[@]}"; do
                        if [ "$select_position" = "top_k_gumbel" ]; then
                            for k in "${ks[@]}"; do
                                for gn in "${gumbel_noise_coefficients[@]}"; do
                                    for unmask_token in "${unmask_tokens[@]}"; do
                                        echo "$ckpt,mdlm_adaptive_score_select_update,score_mask_position=$score_mask_position select_position=$select_position unmask_token=$unmask_token k=$k gumbel_noise_coefficient=$gn $suffix" >> $combinations_file
                                    done
                                done
                            done
                        else
                            for unmask_token in "${unmask_tokens[@]}"; do
                                echo "$ckpt,mdlm_adaptive_score_select_update,score_mask_position=$score_mask_position select_position=$select_position unmask_token=$unmask_token $suffix" >> $combinations_file
                            done
                        fi
                    done
                done
            fi

        # -------------------- GIDD --------------------
        elif [ "$proc" = "gidd" ]; then
            # ---------- p_u unleveraged ----------

            # gidd_emulate_mdlm_vanilla # unmasking only, should be equivalent to gidd_independent_positions_decomposed_update_distribution if p_u=0
            # For each position independently decide to unmask the token with probability p_unmask,
            # p_unmask depends only on the time step and is thus the same for all positions.
            # Parameters:
            # - The strategy to select the new token once a position decides to unmask the current token.
            # Note: unmasked tokens never change, thus the stopping criterion is that all tokens are unmasked.
            if [[ " ${strategies[@]} " =~ " gidd_emulate_mdlm_vanilla " ]]; then
                if (( $(echo "$noise == 0" | bc -l) )); then
                    for unmask_token in "${unmask_tokens[@]}"; do
                        echo "$ckpt,gidd_emulate_mdlm_vanilla,unmask_token=$unmask_token $suffix" >> $combinations_file
                    done
                fi
            fi

            # gidd_emulate_mdlm_adaptive_score_select_update
            # For each position compute a score based on the model's prediction.
            # Based on the scores of the positions, select a subset of positions to update.
            # Update the selected positions according to your strategy.
            # Parameters:
            # - The function to compute the score for each position.
            # - The strategy to select the positions to update based on the scores.
            # - The strategy to select the new token once a position is selected.
            # Note: current implementation does not allow tokens to change once unmasked (designed for p_u=0, should be equivalent to mdlm_adaptive_score_select_update)
            if [[ " ${strategies[@]} " =~ " gidd_emulate_mdlm_adaptive_score_select_update " ]]; then
                if (( $(echo "$noise == 0" | bc -l) )); then
                    for score_mask_position in "${score_mask_positions[@]}"; do
                        for select_position in "${select_positions[@]}"; do
                            if [ "$select_position" = "top_k_gumbel" ]; then
                                for k in "${ks[@]}"; do
                                    for gn in "${gumbel_noise_coefficients[@]}"; do
                                        for unmask_token in "${unmask_tokens[@]}"; do
                                            if [[ "$unmask_token" == "MDM_max" && $(echo "$noise > 0" | bc -l) -eq 1 ]]; then
                                                for self_correction_strategy in "${self_correction_strategies[@]}"; do
                                                    echo "$ckpt,gidd_emulate_mdlm_adaptive_score_select_update,score_mask_position=$score_mask_position select_position=$select_position unmask_token=$unmask_token k=$k gumbel_noise_coefficient=$gn self_correction=$self_correction_strategy $suffix" >> $combinations_file
                                                done
                                            else # Only try self-correction on promising configurations
                                                echo "$ckpt,gidd_emulate_mdlm_adaptive_score_select_update,score_mask_position=$score_mask_position select_position=$select_position unmask_token=$unmask_token k=$k gumbel_noise_coefficient=$gn $suffix" >> $combinations_file
                                            fi
                                        done
                                    done
                                done
                            fi
                        done
                    done
                fi
            fi

            # ---------- p_u leveraged ----------

            # gidd_original # should be equivalent to gidd_independent_positions_decomposed_update_distribution if the token is sampled categorially, but not if max is used (except that here, unmasked tokens can change back to masks)
            # Every position is updated in every step as follows:
            # For each position, compute the probability p_i that the new token is token i, for all i in the vocabulary. And then sample categorically from this distribution.
            # Note: the mask token is also in the vocabulary, thus unmasked tokens can change back to masks. Also, tokens can very well stay the same in a step, according to the probability for that.
            # TODO: check whether unmasked tokens can actually change back to masks
            if [[ " ${strategies[@]} " =~ " gidd_original " ]]; then
                if (( $(echo "$noise > 0" | bc -l) )); then
                    for self_correction_strategy in "${self_correction_strategies[@]}"; do
                        echo "$ckpt,gidd_original,self_correction=$self_correction_strategy $suffix" >> $combinations_file
                    done
                fi
            fi
            
            # gidd_independent_positions_decomposed_update_distribution # if p_u=0, should be equivalent to gidd_emulate_mdlm_vanilla
            # For each position independently decide to change the token with probability p_change,
            # p_change depends on the model's prediction and the current token value and can thus vary from position to position.
            # Parameters:
            # - The strategy to select the new token once a position decides to change the current token.
            # - The stopping criterion (e.g. no masks left)
            # Note: unmasked tokens can change into other tokens, but never back to masks
            if [[ " ${strategies[@]} " =~ " gidd_independent_positions_decomposed_update_distribution " ]]; then
                if (( $(echo "$noise > 0" | bc -l) )); then
                    for change_token in "${change_tokens[@]}"; do
                        echo "$ckpt,gidd_independent_positions_decomposed_update_distribution,change_token=$change_token $suffix" >> $combinations_file
                    done
                fi
            fi

            # gidd_selected_positions_decomposed_update_distribution
            # For each position compute the probability to change the token based on the model's prediction.
            # Also compute the probability to unmask at the current time step. This does not depend on the position.
            # Check if there is at least one position where the probability to change is greater than the probability to unmask.
            # If so, change a subset of those positions according to your "change_strategy". The subset is selected according to the "select_position_change" strategy.
            # If not, choose a subset of currently masked positions and unmask them according to your "unmask_strategy". The subset is selected according to the "select_position_unmask" strategy.
            # Parameters:
            # - "change_strategy", composed of a strategy to select subset of the eligible positions to change, and strategy to select the new token once a position is selected.
            # - "unmask_strategy", potentially composed of a score function for each masked position, strategy to select the positions to unmask, and strategy to select the new token once a position is selected.
            # Note: the "change_strategy" should not allow unmasked tokens to turn back into masks.
            if [[ " ${strategies[@]} " =~ " gidd_selected_positions_decomposed_update_distribution " ]]; then
                if (( $(echo "$noise > 0" | bc -l) )); then
                    for score_mask_position in "${score_mask_positions[@]}"; do
                        for sel_change in "${select_positions_to_change[@]}"; do
                            for sel_unmask in "${select_positions[@]}"; do
                                if [ "$sel_unmask" = "top_k_gumbel" ]; then
                                    for k in "${ks[@]}"; do
                                        for gn in "${gumbel_noise_coefficients[@]}"; do
                                            for change_token in "${change_tokens[@]}"; do
                                                for unmask_token in "${unmask_tokens[@]}"; do
                                                    echo "$ckpt,gidd_selected_positions_decomposed_update_distribution,score_mask_position=$score_mask_position select_position_change=$sel_change select_position_unmask=$sel_unmask change_token=$change_token unmask_token=$unmask_token k=$k gumbel_noise_coefficient=$gn $suffix" >> $combinations_file
                                                done
                                            done
                                        done
                                    done
                                else
                                    for change_token in "${change_tokens[@]}"; do
                                        for unmask_token in "${unmask_tokens[@]}"; do
                                            echo "$ckpt,gidd_selected_positions_decomposed_update_distribution,score_mask_position=$score_mask_position select_position_change=$sel_change select_position_unmask=$sel_unmask change_token=$change_token unmask_token=$unmask_token $suffix" >> $combinations_file
                                        done
                                    done
                                fi
                            done
                        done
                    done
                fi
            fi

            # gidd_change_low_confidence_positions
            if [[ " ${strategies[@]} " =~ " gidd_change_low_confidence_positions " ]]; then
                if (( $(echo "$noise > 0" | bc -l) )); then
                    for score_mask_position in "${score_mask_positions[@]}"; do
                        for select_position_unmask in "${select_positions[@]}"; do
                            if [ "$select_position_unmask" = "top_k_gumbel" ]; then
                                for k in "${ks[@]}"; do
                                    for gn in "${gumbel_noise_coefficients[@]}"; do
                                        for change_token in "${change_tokens[@]}"; do
                                            for unmask_token in "${unmask_tokens[@]}"; do
                                                echo "$ckpt,gidd_change_low_confidence_positions,score_mask_position=$score_mask_position select_position_unmask=$select_position_unmask change_token=$change_token unmask_token=$unmask_token k=$k gumbel_noise_coefficient=$gn $suffix" >> $combinations_file
                                            done
                                        done
                                    done
                                done
                            fi
                        done
                    done
                fi
            fi

            # gidd_change_based_on_model_confidence_to_change
            # For each position, compute the score as confidence of the model on any token that is different from the current token.
            # Based on the scores of the positions, select a subset of positions to update.
            # Update the selected positions according to your strategy.
            # Parameters:
            # - The function to compute the score for each position.
            # - The strategy to select the positions to update based on the scores.
            # - The strategy to select the new token once a position is selected.
            if [[ " ${strategies[@]} " =~ " gidd_change_based_on_model_confidence_to_change " ]]; then
                if (( $(echo "$noise > 0" | bc -l) )); then
                    for score_position_for_change in "${score_positions_for_change[@]}"; do
                        for select_position in "${select_positions[@]}"; do
                            if [ "$select_position" = "top_k_gumbel" ]; then
                                for k in "${ks[@]}"; do
                                    for gn in "${gumbel_noise_coefficients[@]}"; do
                                        for change_token in "${change_tokens[@]}"; do
                                            if [[ "$change_token" == "change_max" && $(echo "$noise > 0" | bc -l) -eq 1 ]]; then
                                                for self_correction_strategy in "${self_correction_strategies[@]}"; do
                                                    echo "$ckpt,gidd_change_based_on_model_confidence_to_change,score_position_for_change=$score_position_for_change select_position=$select_position change_token=$change_token k=$k gumbel_noise_coefficient=$gn self_correction=$self_correction_strategy $suffix" >> $combinations_file
                                                done
                                            else # Only try self-correction on promising configurations
                                                echo "$ckpt,gidd_change_based_on_model_confidence_to_change,score_position_for_change=$score_position_for_change select_position=$select_position change_token=$change_token k=$k gumbel_noise_coefficient=$gn $suffix" >> $combinations_file
                                            fi
                                        done
                                    done
                                done
                            fi
                        done
                    done
                fi
            fi
            
            # gidd_keep_where_confident
            # For each position, compute the score as either:
            # - 0 if the current token is the most likely token according to the model's prediction.
            # - The confidence/margin of the model otherwise.
            # Based on the scores of the positions, select a subset of positions to update.
            # Update the selected positions according to your strategy.
            # Parameters:
            # - The function to compute the score for each position (confidence or margin).
            # - The strategy to select the positions to update based on the scores.
            # - The strategy to select the new token once a position is selected.
            if [[ " ${strategies[@]} " =~ " gidd_keep_where_confident " ]]; then
                if (( $(echo "$noise > 0" | bc -l) )); then
                    for score_position_for_change in "${score_positions_for_change[@]}"; do
                        for select_position in "${select_positions[@]}"; do
                            if [ "$select_position" = "top_k_gumbel" ]; then
                                for k in "${ks[@]}"; do
                                    for gn in "${gumbel_noise_coefficients[@]}"; do
                                        for change_token in "${change_tokens[@]}"; do
                                            if [[ "$change_token" == "change_max" && $(echo "$noise > 0" | bc -l) -eq 1 ]]; then
                                                for self_correction_strategy in "${self_correction_strategies[@]}"; do
                                                    echo "$ckpt,gidd_keep_where_confident,score_position_for_change=$score_position_for_change select_position=$select_position change_token=$change_token k=$k gumbel_noise_coefficient=$gn self_correction=$self_correction_strategy $suffix" >> $combinations_file
                                                done
                                            else # Only try self-correction on promising configurations
                                                echo "$ckpt,gidd_keep_where_confident,score_position_for_change=$score_position_for_change select_position=$select_position change_token=$change_token k=$k gumbel_noise_coefficient=$gn $suffix" >> $combinations_file
                                            fi
                                        done
                                    done
                                done
                            fi
                        done
                    done
                fi
            fi

            # gidd_flattened
            if [[ " ${strategies[@]} " =~ " gidd_flattened " ]]; then
                if (( $(echo "$noise > 0" | bc -l) )); then
                    echo "$ckpt,gidd_flattened,$suffix" >> $combinations_file
                fi
            fi

            # gidd_prob_to_recover_data
            if [[ " ${strategies[@]} " =~ " gidd_prob_to_recover_data " ]]; then
                if (( $(echo "$noise > 0" | bc -l) )); then
                    for oracle in "${oracles[@]}"; do
                        for position_sampling in "${position_sampling_strategies[@]}"; do
                            for position_metric in "${position_metric_strategies[@]}"; do
                                for token_sampling in "${token_sampling_strategies[@]}"; do
                                    if [[ "$position_sampling" == "independent" && "$position_metric" == "p_denoise" ]]; then
                                        for uniform_noise in "${uniform_noise_strategies[@]}"; do
                                            echo "$ckpt,gidd_prob_to_recover_data,oracle=$oracle position_sampling=$position_sampling position_metric=$position_metric token_sampling=$token_sampling uniform_noise=$uniform_noise $suffix" >> $combinations_file
                                        done
                                    else
                                        if [[ "$oracle" == "model_EMA" && "$position_sampling" == "independent" && "$position_metric" == "confident_and_noisy" && "$token_sampling" == "max" ]]; then
                                            # try adding self-correction
                                            for self_correction_strategy in "${self_correction_strategies[@]}"; do
                                                echo "$ckpt,gidd_prob_to_recover_data,oracle=$oracle position_sampling=$position_sampling position_metric=$position_metric token_sampling=$token_sampling self_correction=$self_correction_strategy $suffix" >> $combinations_file
                                            done
                                        else
                                            if [[ "$position_sampling" == "top_k" ]]; then
                                                for k in "${ks[@]}"; do
                                                    echo "$ckpt,gidd_prob_to_recover_data,oracle=$oracle position_sampling=$position_sampling position_metric=$position_metric token_sampling=$token_sampling k=$k $suffix" >> $combinations_file
                                                done
                                            else
                                                echo "$ckpt,gidd_prob_to_recover_data,oracle=$oracle position_sampling=$position_sampling position_metric=$position_metric token_sampling=$token_sampling $suffix" >> $combinations_file
                                            fi
                                        fi
                                    fi
                                done
                            done
                        done
                    done
                fi
            fi
        fi
    done
}

build_combinations_file=true
if [ $build_combinations_file = true ]; then
    > $combinations_file
    for seed in "${seeds[@]}"; do
        for dataset in "${datasets[@]}"; do
            for num_denoising_steps in "${nums_denoising_steps[@]}"; do
                for num_self_correction_steps in "${nums_self_correction_steps[@]}"; do
                    for do_beam_search in "${beam_search[@]}"; do
                        if [ "$do_beam_search" = "true" ]; then
                            current_batch_size=1
                            general_suffix="dataset=${dataset} num_samples=$num_samples num_denoising_steps=$num_denoising_steps num_self_correction_steps=$num_self_correction_steps batch_size=$current_batch_size min_p=$min_p compile_torch=$compile_torch seed=$seed"
                            for num_steps_before_pruning in "${steps_before_pruning[@]}"; do
                                for num_pruning_beams in "${pruning_num_beams[@]}"; do
                                    for branching_factor in "${branching_factors[@]}"; do
                                        for score_time in "${score_times[@]}"; do
                                            for score_method in "${score_methods[@]}"; do
                                                for initiate_after in "${initiate_beam_search_after_progress[@]}"; do
                                                    beam_search_suffix="do_beam_search=true steps_before_pruning=$num_steps_before_pruning pruning_num_beams=$num_pruning_beams branching_factor=$branching_factor score_time=$score_time score_method=$score_method initiate_beam_search_after_progress=$initiate_after"
                                                    suffix="$general_suffix $beam_search_suffix"
                                                    process_checkpoints "$suffix"
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        else
                            current_batch_size=$batch_size
                            general_suffix="dataset=${dataset} num_samples=$num_samples num_denoising_steps=$num_denoising_steps num_self_correction_steps=$num_self_correction_steps batch_size=$current_batch_size min_p=$min_p compile_torch=$compile_torch seed=$seed"
                            beam_search_suffix="do_beam_search=false"
                            suffix="$general_suffix $beam_search_suffix"
                            process_checkpoints "$suffix"
                        fi
                    done
                done
            done
        done
    done
fi
# cat $combinations_file

echo "accuracy,correctly_filled_cells,not_fully_unmasked,checkpoint,strategy,params,true_denoised_fraction,denoised_fraction,mask_fraction,uniform_fraction,history" > $csv_file

evaluate_one_py="$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )/evaluate_one.py"
export evaluate_one_py
export csv_file
VENV_ACTIVATE_PATH="$( dirname "${BASH_SOURCE[0]}" )/../../.venv/bin/activate"
source $VENV_ACTIVATE_PATH

gpu_locking_dir="$( dirname "${BASH_SOURCE[0]}" )/../.."
gpu_locking_dir="$( cd $gpu_locking_dir && pwd )"
acquire_gpu_path="$gpu_locking_dir/aquire_gpu.sh"
release_gpu_path="$gpu_locking_dir/release_gpu.sh"
export acquire_gpu_path
export release_gpu_path

clear_gpu_locks_path="$gpu_locking_dir/clear_gpu_locks.sh"
bash "$clear_gpu_locks_path"

echo "Starting evaluation of combinations from $combinations_file"
cat $combinations_file | parallel --colsep ',' -j $num_jobs '
    GPU=$(bash "$acquire_gpu_path")

    CKPT={1}
    STRATEGY={2}
    PARAMS_ORIGINAL={3}
    PARAMS=$(echo "$PARAMS_ORIGINAL" | sed "s/\([^ =]*=[^ ]*\)/--\1/g")

    # CMD="python3 $evaluate_one_py --checkpoint $CKPT --strategy $STRATEGY $PARAMS"
    CMD="python3 $evaluate_one_py --checkpoint $CKPT --strategy $STRATEGY $PARAMS --combinations_row=$(({#}))"
    CMD="$CMD --device=$GPU"
    # echo "Running on GPU $GPU: $CMD"
    echo "Running line $(({#}))"
    OUTPUT=$(CUDA_VISIBLE_DEVICES=$GPU $CMD)
    ACCURACY=$(echo "$OUTPUT" | grep -oP "accuracy=\K[0-9.]+")
    CORRECTLY_FILLED_CELLS=$(echo "$OUTPUT" | grep -oP "correctly_filled_cells=\K[0-9.]+")
    NOT_FULLY_UNMASKED=$(echo "$OUTPUT" | grep -oP "not_fully_unmasked=\K[0-9.]+")
    HISTORY=$(echo "$OUTPUT" | grep -oP "\"num_steps=[0-9]+ seq_len=[0-9]+ history=[0-9 ]+\"")
    TRUE_DENOISED_FRACTION=$(echo "$OUTPUT" | grep -oP "\"true_denoised_fraction=[^\"]+\"")
    DENOISED_FRACTION=$(echo "$OUTPUT" | grep -oP "\"denoised_fraction=[^\"]+\"")
    MASK_FRACTION=$(echo "$OUTPUT" | grep -oP "\"mask_fraction=[^\"]+\"")
    UNIFORM_FRACTION=$(echo "$OUTPUT" | grep -oP "\"uniform_fraction=[^\"]+\"")
    # echo "$ACCURACY,$CORRECTLY_FILLED_CELLS,$NOT_FULLY_UNMASKED,$CKPT,$STRATEGY,\"$PARAMS_ORIGINAL\",$HISTORY" >> $csv_file
    echo "$ACCURACY,$CORRECTLY_FILLED_CELLS,$NOT_FULLY_UNMASKED,$CKPT,$STRATEGY,\"$PARAMS_ORIGINAL\",$TRUE_DENOISED_FRACTION,$DENOISED_FRACTION,$MASK_FRACTION,$UNIFORM_FRACTION,$HISTORY" >> $csv_file
    echo "Output for line $(({#})):"
    echo "$OUTPUT"

    bash "$release_gpu_path" $GPU
'

aggregate_results_over_seeds_py="$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )/aggregate_results_over_seeds.py"
python3 $aggregate_results_over_seeds_py --csv_file $csv_file