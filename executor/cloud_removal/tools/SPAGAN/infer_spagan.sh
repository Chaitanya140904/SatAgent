#!/bin/bash

# Define the base directory to avoid repetition
BASE_DIR="/home/project/satagent/satagent_impl/executor/cloud_removal/tools/SPAGAN"

python $BASE_DIR/infer_spagan.py \
  --config "$BASE_DIR/SpA-GAN_for_cloud_removal/config.yml" \
  --input_path "/home/project/satagent/satagent_impl/executor/cloud_removal/test_images/test_image.jpg" \
  --output_path "$BASE_DIR/output.png" \
  --pretrained "$BASE_DIR/SpA-GAN_for_cloud_removal/pretrained_models/RICE1/gen_model_epoch_200.pth" \
  --cuda \
  --gpu_ids 0