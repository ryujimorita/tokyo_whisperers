#!/bin/bash

# Whisper Tiny evaluations
poetry run python run_evaluation.py \
    --model_type whisper \
    --model_path "openai/whisper-tiny" \
    --dataset_config "conf/dataset_config_no_cv.yaml" \
    --output_file "results/baseline_tiny.json"

poetry run python run_evaluation.py \
    --model_type whisper \
    --model_path "output/e2e_spec_aug" \
    --dataset_config "conf/dataset_config_no_cv.yaml" \
    --output_file "results/e2e_tiny.json"

poetry run python run_evaluation.py \
    --model_type whisper \
    --model_path "output/r_254_a_128_d_025_spec_aug" \
    --dataset_config "conf/dataset_config_no_cv.yaml" \
    --lora_path "output/r_254_a_128_d_025_spec_aug" \
    --output_file "results/lora_tiny.json"

# Whisper Base evaluations
poetry run python run_evaluation.py \
    --model_type whisper \
    --model_path "openai/whisper-base" \
    --dataset_config "conf/dataset_config_no_cv.yaml" \
    --output_file "results/baseline_base.json"

poetry run python run_evaluation.py \
    --model_type whisper \
    --model_path "output/e2e_spec_aug_base" \
    --dataset_config "conf/dataset_config_no_cv.yaml" \
    --output_file "results/e2e_base.json"

poetry run python run_evaluation.py \
    --model_type whisper \
    --model_path "output/r_254_a_128_d_025_spec_aug_base" \
    --dataset_config "conf/dataset_config_no_cv.yaml" \
    --lora_path "output/r_254_a_128_d_025_spec_aug_base" \
    --output_file "results/lora_base.json"

# Whisper Small evaluations
poetry run python run_evaluation.py \
    --model_type whisper \
    --model_path "openai/whisper-small" \
    --dataset_config "conf/dataset_config_no_cv.yaml" \
    --output_file "results/baseline_small.json"

poetry run python run_evaluation.py \
    --model_type whisper \
    --model_path "output/e2e_spec_aug_small" \
    --dataset_config "conf/dataset_config_no_cv.yaml" \
    --output_file "results/e2e_small.json"

poetry run python run_evaluation.py \
    --model_type whisper \
    --model_path "output/r_254_a_128_d_025_spec_aug_small" \
    --dataset_config "conf/dataset_config_no_cv.yaml" \
    --lora_path "output/r_254_a_128_d_025_spec_aug_small" \
    --output_file "results/lora_small.json"

poetry run python run_evaluation.py \
    --model_type whisper \
    --model_path "openai/whisper-medium" \
    --dataset_config "conf/dataset_config_no_cv.yaml" \
    --output_file "results/baseline_medium.json"

poetry run python run_evaluation.py \
    --model_type whisper \
    --model_path "openai/whisper-large-v3" \
    --dataset_config "conf/dataset_config_no_cv.yaml" \
    --output_file "results/baseline_large_v3.json"

# ReazonSpeech evaluations
poetry run python run_evaluation.py \
    --model_type k2 \
    --dataset_config "conf/dataset_config_no_cv.yaml" \
    --output_file "results/reazonspeech_k2.json"

poetry run python run_evaluation.py \
    --model_type nemo \
    --dataset_config "conf/dataset_config_no_cv.yaml" \
    --output_file "results/reazonspeech_nemo.json"

poetry run python run_evaluation.py \
    --model_type espnet \
    --dataset_config "conf/dataset_config_no_cv.yaml" \
    --output_file "results/reazonspeech_espnet.json"

# echo "Running ReazonSpeech OneSeg evaluation..."
# poetry run python run_evaluation.py \
#     --model_type oneseg \
#     --dataset_config "conf/dataset_config_no_cv.yaml" \
#     --output_file "results/reazonspeech_oneseg.json"