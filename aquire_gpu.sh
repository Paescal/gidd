#!/bin/bash
AVAILABLE_GPUS=(0 1)

LOCK_DIR="$( dirname "${BASH_SOURCE[0]}" )"
LOCK_DIR="$( cd $LOCK_DIR && pwd )/tmp/gpu_locks"
mkdir -p "$LOCK_DIR"

while true; do
    for GPU in "${AVAILABLE_GPUS[@]}"; do
        LOCK_FILE="$LOCK_DIR/gpu_lock_$GPU"

        if ( set -o noclobber; > "$LOCK_FILE" ) 2> /dev/null; then
            echo "$GPU"
            exit 0
        fi
    done
    sleep 2
done
