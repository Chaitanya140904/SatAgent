<!-- The command for running is:

rm -f configs/model_health.json sata/configs/model_health.json

SATA_SWINIR_SR_BACKEND="$(pwd)/executor/artifact_removal/tools/swinIR/SwinIR/main_test_swinir.py" \
SATA_TOOL_CMD_TIMEOUT_SEC=900 \
CUDA_VISIBLE_DEVICES=1 \
python3 main.py \
  --image datasets/UCMerced/UCMerced_LandUse/Images/agricultural/agricultural40.tif \
  --profile configs/profiles/Aer4K_P.yaml \
  --output-dir outputs/final_production_run \
  --strict-real-execution \
  --require-true-models \
  --tool-run-gpu-id 1

SATA_SWINIR_SR_BACKEND="$(pwd)/executor/artifact_removal/tools/swinIR/SwinIR/main_test_swinir.py" \
SATA_TOOL_CMD_TIMEOUT_SEC=900 \
CUDA_VISIBLE_DEVICES=1 \
python3 main.py \
  --image satimg.jpeg \
  --profile configs/profiles/Aer4K_P.yaml \
  --output-dir outputs/final_production_run \
  --strict-real-execution \
  --require-true-models \
  --tool-run-gpu-id 1 -->

  find . -type d -name "__pycache__" -exec rm -rf {} +

SATA_SWINIR_SR_BACKEND="$(pwd)/executor/artifact_removal/tools/swinIR/SwinIR/main_test_swinir.py" \
SATA_TOOL_CMD_TIMEOUT_SEC=900 \
CUDA_VISIBLE_DEVICES=1 \
python3 main.py \
  --image degraded_image.jpg \
  --profile configs/profiles/Aer4K_P.yaml \
  --output-dir outputs/perception_restoration \
  --strict-real-execution \
  --tool-run-gpu-id 1

  find . -type d -name "__pycache__" -exec rm -rf {} +

SATA_SWINIR_SR_BACKEND="$(pwd)/executor/artifact_removal/tools/swinIR/SwinIR/main_test_swinir.py" \
SATA_TOOL_CMD_TIMEOUT_SEC=900 \
CUDA_VISIBLE_DEVICES=1 \
python3 main.py \
  --image satimg.jpeg \
  --profile configs/profiles/Aer4K_H.yaml \
  --output-dir outputs/hybrid_restoration \
  --strict-real-execution \
  --tool-run-gpu-id 1

  find . -type d -name "__pycache__" -exec rm -rf {} +

SATA_SWINIR_SR_BACKEND="$(pwd)/executor/artifact_removal/tools/swinIR/SwinIR/main_test_swinir.py" \
SATA_TOOL_CMD_TIMEOUT_SEC=900 \
CUDA_VISIBLE_DEVICES=1 \
python3 main.py \
  --image satimg.jpeg \
  --profile configs/profiles/Aer4K_F.yaml \
  --output-dir outputs/fidelity_restoration \
  --strict-real-execution \
  --tool-run-gpu-id 1
