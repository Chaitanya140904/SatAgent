#!/bin/bash
# infer_transenet.sh

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate satagent # Or your TransENet environment

export CUDA_VISIBLE_DEVICES=1

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# Usage: ./infer_transenet.sh input.png output.png 4
python "$SCRIPT_DIR/infer_transenet.py" \
    --input_path "${1:-../../test_images/image.png}" \
    --output_path "${2:-./transenet_output.png}" \
    --scale "${3:-4}"