#!/bin/bash

LOCK_DIR="$( dirname "${BASH_SOURCE[0]}" )"
LOCK_DIR="$( cd $LOCK_DIR && pwd )/tmp/gpu_locks"
GPU=$1
rm -f "$LOCK_DIR/gpu_lock_$GPU"