poetry run python3 main.py \
    --model_name_or_path="openai/whisper-tiny" \
    --dataset_config_path="conf/dataset_config.yaml" \
    --wandb_run_name="mark_tiny_with_reg_full_dataset_fleurs_only_v2" \
    --train_dataset_fraction=1 \
    --eval_dataset_fraction=1 \
    --dataset_config_name="ja" \
    --language="japanese" \
    --max_steps="10000" \
    --output_dir="./output/reg_full_dataset_fleurs_only_v2" \
	--save_total_limit="5" \
	--per_device_train_batch_size="8" \
	--per_device_eval_batch_size="8" \
    --gradient_accumulation_steps="2" \
	--logging_steps="5" \
	--learning_rate="1e-7" \
	--warmup_steps="100" \
	--evaluation_strategy="steps" \
	--eval_steps="5" \
	--save_strategy="steps" \
	--save_steps="5" \
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
	--apply_spec_augment \
	--load_best_model_at_end \
	--gradient_checkpointing \
	--fp16 \
	--overwrite_output_dir \
	--do_train \
	--do_eval \
	--predict_with_generate \
	--use_auth_token