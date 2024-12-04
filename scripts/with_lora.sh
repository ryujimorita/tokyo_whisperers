poetry run python3 main.py \
    --model_name_or_path="openai/whisper-tiny" \
    --dataset_config_path="conf/dataset_config.yaml" \
    --wandb_run_name="with_lora_v1" \
    --train_dataset_fraction=0.1 \
    --eval_dataset_fraction=0.1 \
    --dataset_config_name="ja" \
    --language="japanese" \
    --max_steps="50" \
    --output_dir="./output/with_lora" \
    --save_total_limit="3" \
    --per_device_train_batch_size="8" \
    --per_device_eval_batch_size="8" \
    --gradient_accumulation_steps="2" \
    --logging_steps="5" \
    --learning_rate="1e-3" \
    --warmup_steps="0" \
    --evaluation_strategy="steps" \
    --eval_steps="5" \
    --save_strategy="steps" \
    --save_steps="5" \
    --use_lora \
    --lora_r="8" \
    --lora_alpha="16" \
    --lora_dropout="0.3" \
    --lora_target_modules="q_proj,v_proj" \
    --weight_decay="0.01" \
    --dropout="0.1" \
    --attention_dropout="0.1" \
    --activation_dropout="0.1" \
    --load_best_model_at_end \
    --gradient_checkpointing \
    --fp16 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --predict_with_generate \
    --use_auth_token