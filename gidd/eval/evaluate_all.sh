#!/bin/bash

checkpoints=(
    "2025-06-18/12-51-55 gidd 0"
    "2025-06-13/15-04-32 gidd 0.2"
    "2025-06-11/18-28-57 mdlm -"
    "2025-06-01/17-31-45 gidd 0"
)

score_positions=(
    "MDM_max"
    "MDM_margin"
)
select_positions=(
    "all"
    "independent"
    "top_k_gumbel"
)
update_tokens=(
    "MDM_max"
    "MDM_categorical"
    "categorical"
)


> combinations.txt
for entry in "${checkpoints[@]}"; do
    ckpt=$(echo $entry | awk '{print $1}')
    proc=$(echo $entry | awk '{print $2}')
    noise=$(echo $entry | awk '{print $3}')

    if [ "$proc" = "mdlm" ]; then
        # mdlm_vanilla
        for ut in "${update_tokens[@]}"; do
            echo "$ckpt mdlm_vanilla update_token=$ut" >> combinations.txt
        done
        
        # mdlm_adaptive_score_select_update
        for sp in "${score_positions[@]}"; do
            for sel in "${select_positions[@]}"; do
                for ut in "${update_tokens[@]}"; do
                    echo "$ckpt mdlm_adaptive_score_select_update update_token=$ut score_position=$sp select_position=$sel" >> combinations.txt
                done
            done
        done

    elif [ "$proc" = "gidd" ]; then
        # gidd_vanilla_original
        echo "$ckpt gidd_vanilla_original" >> combinations.txt

        # gidd_vanilla_split
        for ut in "${update_tokens[@]}"; do
            echo "$ckpt gidd_vanilla_split update_token=$ut" >> combinations.txt
        done
        
        # gidd_adaptive_score_select_update
        for sp in "${score_positions[@]}"; do
            for sel in "${select_positions[@]}"; do
                for ut in "${update_tokens[@]}"; do
                    echo "$ckpt gidd_adaptive_score_select_update update_token=$ut score_position=$sp select_position=$sel" >> combinations.txt
                done
            done
        done

        # gidd_adaptive_change_vs_unmask
        if (( $(echo "$noise > 0" | bc -l) )); then
            for sp in "${score_positions[@]}"; do
                for sel in "${select_positions[@]}"; do
                    for ut in "${update_tokens[@]}"; do
                        echo "$ckpt gidd_adaptive_change_vs_unmask update_token=$ut score_position=$sp select_position=$sel" >> combinations.txt
                    done
                done
            done
        fi
    fi
done

cat combinations.txt

cat combinations.txt | parallel --colsep ' ' -j 4 '
  CKPT={1}
  STRATEGY={2}
  shift 2

  CMD="python evaluate_one.py --checkpoint $CKPT --strategy $STRATEGY"

  for kv in {=3..}; do
    KEY=$(echo $kv | cut -d= -f1)
    VAL=$(echo $kv | cut -d= -f2)
    if [ -n "$KEY" ] && [ -n "$VAL" ]; then
      CMD="$CMD --$KEY $VAL"
    fi
  done

  echo "Running: $CMD"
  $CMD
'