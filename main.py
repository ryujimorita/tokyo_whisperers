import os
import sys
import logging
from transformers import (
    HfArgumentParser, 
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from datasets import DatasetDict
from schemas import ModelArguments, DataTrainingArguments, DataCollatorSpeechSeq2SeqWithPadding
from src.dataloader import load_datasets_from_config
from src.metrics import MetricsCalculator, TextNormalizer
from src.callbacks import ShuffleCallback, EpochProgressCallback
from loguru import logger

def main():
    # parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    # setting seed for reproducibility
    set_seed(training_args.seed)
    
    # load datasets
    raw_datasets = DatasetDict()
    if training_args.do_train:
        raw_datasets["train"] = load_datasets_from_config(
            "conf/dataset_config.yaml",
            "train",
            16000
        )
    
    if training_args.do_eval:
        raw_datasets["eval"] = load_datasets_from_config(
            "conf/dataset_config.yaml", 
            "eval",
            16000
        )

    # load model and tokenizer
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_args.model_name_or_path)

    # settup model configuration
    model.config.language = "japanese"
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()
    if model_args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False
        
    # init processor and data collator
    feature_extractor.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    config.save_pretrained(training_args.output_dir)
    processor = AutoProcessor.from_pretrained(training_args.output_dir)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # init text normalizer
    text_normalizer = TextNormalizer()

    # preproc datasets -> sampling to 16k
    def prepare_dataset(batch):
        # process audio
        sample = batch[data_args.audio_column_name]
        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        batch[feature_extractor.model_input_names[0]] = inputs.get(feature_extractor.model_input_names[0])[0]
        batch["input_length"] = len(sample["array"])

        # process text 
        # TODO: check manually the labels could be weird
        input_str = batch[data_args.text_column_name]
        input_str = text_normalizer.normalize(
            input_str, 
            do_lower=data_args.do_lower_case
        )
        if data_args.do_remove_punctuation:
            input_str = text_normalizer.normalizer(input_str).strip()
        
        batch["labels"] = tokenizer(input_str).input_ids
        return batch

    # process datasets
    vectorized_datasets = raw_datasets.map(
        prepare_dataset,
        remove_columns=next(iter(raw_datasets.values())).features.keys(),
    ).with_format("torch")

    if training_args.do_train:
        vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(
            # buffer_size=data_args.shuffle_buffer_size,
            seed=training_args.seed,
        )

    # filter datasets based on audio length
    max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate

    def is_audio_in_length_range(length):
        return min_input_length < length < max_input_length

    vectorized_datasets["train"] = vectorized_datasets["train"].filter(
        is_audio_in_length_range,
        input_columns=["input_length"],
    )

    # init metrics calculator
    metrics_calculator = MetricsCalculator(
        tokenizer=tokenizer,
        do_normalize_eval=data_args.do_normalize_eval
    )

    # save processor components
    if is_main_process(training_args.local_rank):
        feature_extractor.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)

    # init trainer
    # TODO: add regularization
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
        tokenizer=feature_extractor,
        data_collator=data_collator,
        compute_metrics=metrics_calculator.compute_metrics if training_args.predict_with_generate else None,
        callbacks=[ShuffleCallback()],
    )

    # training
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        # log and save metrics
        # TODO: add json metrics saver for plotting later
        metrics = train_result.metrics
        if data_args.max_train_samples:
            metrics["train_samples"] = data_args.max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # eval
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
        )
        
        if data_args.max_eval_samples:
            metrics["eval_samples"] = data_args.max_eval_samples

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write Training Stats
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "automatic-speech-recognition",
        "tags": "whisper-event",
    }

    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name
        if "common_voice" in data_args.dataset_name:
            kwargs["language"] = data_args.dataset_config_name
        if model_args.model_index_name is not None:
            kwargs["model_name"] = model_args.model_index_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results

if __name__ == "__main__":
    main()