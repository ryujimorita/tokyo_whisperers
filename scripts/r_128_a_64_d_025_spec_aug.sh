poetry run python3 main.py \
    --model_name_or_path="openai/whisper-tiny" \
    --dataset_config_path="conf/dataset_config_no_cv.yaml" \
    --wandb_run_name="r_128_a_64_d_025_spec_aug" \
    --train_dataset_fraction=1 \
    --eval_dataset_fraction=0.1 \
    --dataset_config_name="ja" \
    --language="japanese" \
    --max_steps="5000" \
    --output_dir="./output/r_128_a_64_d_025_spec_aug" \
    --save_total_limit="3" \
	--per_device_train_batch_size="32" \
	--per_device_eval_batch_size="32" \
    --gradient_accumulation_steps="1" \
	--logging_steps="5" \
	--learning_rate="1e-4" \
	--warmup_steps="100" \
	--evaluation_strategy="steps" \
	--eval_steps="50" \
	--save_strategy="steps" \
	--save_steps="50" \
	--generation_max_length="225" \
	--length_column_name="input_length" \
	--max_duration_in_seconds="30" \
	--text_column_name="sentence" \
	--freeze_feature_encoder="False" \
	--report_to="wandb" \
	--metric_for_best_model="wer" \
	--greater_is_better="False" \
	--weight_decay="0.3" \
	--dropout="0.2" \
	--attention_dropout="0.2" \
	--activation_dropout="0.2" \
	--load_best_model_at_end \
	--gradient_checkpointing \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--use_auth_token \
    --use_lora \
    --lora_r="128" \
    --lora_alpha="64" \
    --lora_dropout="0.25" \
    --lora_target_modules="q_proj,v_proj" \
	--apply_spec_augment