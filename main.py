import os
from dotenv import load_dotenv
import wandb
import sys
import logging
import warnings
import pandas as pd
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
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from datasets import DatasetDict, concatenate_datasets
from schemas import (
    ModelArguments,
    DataTrainingArguments,
    DataCollatorSpeechSeq2SeqWithPadding,
    LoRAArguments,
)
from src.augment import DataAugmentator
from src.dataloader import load_datasets_from_config
from src.metrics import MetricsCalculator, TextNormalizer
from src.callbacks import ShuffleCallback, EpochProgressCallback, SavePeftModelCallback
from loguru import logger
from src.metrics_cache import MetricsCache
from peft import get_peft_model, prepare_model_for_kbit_training

# load environment variables from .env file
load_dotenv()

# init wandb
os.environ["WANDB_PROJECT"] = os.getenv(
    "WANDB_PROJECT", "tokyo_whisperers"
)  # TODO: can get from args?
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

wandb.login(key=os.getenv("WANDB_API_KEY"))


def _suppress_warnings() -> None:
    """Suppresses specific warnings and logs for cleaner output."""
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)


DEBUG = False
if not DEBUG:
    _suppress_warnings()


def main():
    # parse arguments
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, LoRAArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, lora_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, lora_args = (
            parser.parse_args_into_dataclasses()
        )

    # setting seed for reproducibility
    logger.info(f"Setting seed to {training_args.seed}")
    set_seed(training_args.seed)

    # init metrics cache
    logger.info(f"Initializing metrics cache at {training_args.output_dir}")
    mt = MetricsCache(training_args.output_dir)

    # load datasets
    raw_datasets = DatasetDict()
    if training_args.do_train:
        logger.info(f"Loading training dataset from {data_args.dataset_config_path}")
        raw_datasets["train"] = load_datasets_from_config(
            data_args.dataset_config_path,
            "train",
            16000,  # whisper sampling rate
            data_args.train_dataset_fraction,
        )

    if training_args.do_eval:
        logger.info(f"Loading evaluation dataset from {data_args.dataset_config_path}")
        raw_datasets["eval"] = load_datasets_from_config(
            data_args.dataset_config_path,
            "eval",
            16000,
            data_args.eval_dataset_fraction,
        )

    if data_args.do_augment:
        logger.info(
            f"Training data size - before augmentation: {len(raw_datasets['train'])}"
        )
        # init data augmentator
        data_augmentator = DataAugmentator(data_args.audio_column_name)
        # augment training data
        augmented_raw_training_dataset = raw_datasets["train"].map(
            data_augmentator.augment_dataset,
            desc="Applying augmentation to the training dataset",
        )

        # combine original training data and augmented data
        raw_datasets["train"] = concatenate_datasets(
            [raw_datasets["train"], augmented_raw_training_dataset]
        )

        logger.info(
            f"Training data size - after augmentation: {len(raw_datasets['train'])}"
        )

    # load model and tokenizer
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        model_args.model_name_or_path,
        language="japanese",
        task="transcribe",
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        model_args.model_name_or_path,
        language="japanese",
        task="transcribe",
    )
    if hasattr(model_args, "use_lora") and model_args.use_lora:
        logger.info(
            "Loading model in 8-bit mode..."
        )  # TODO: explore not using this shiz
        model = WhisperForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, load_in_8bit=True, device_map="auto"
        )
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path
        )

    # set up model configuration
    model.config.language = "japanese"
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    # https://huggingface.co/docs/transformers/en/model_doc/whisper#transformers.WhisperConfig
    model.config.dropout = (
        training_args.dropout if hasattr(training_args, "dropout") else 0.1
    )
    model.config.attention_dropout = (
        training_args.attention_dropout
        if hasattr(training_args, "attention_dropout")
        else 0.1
    )
    model.config.activation_dropout = (
        training_args.activation_dropout
        if hasattr(training_args, "activation_dropout")
        else 0.1
    )

    # init processor and data collator
    feature_extractor.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    config.save_pretrained(training_args.output_dir)
    processor = WhisperProcessor.from_pretrained(
        training_args.output_dir, language="japanese", task="transcribe"
    )
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
        inputs = feature_extractor(
            sample["array"], sampling_rate=sample["sampling_rate"]
        )
        batch[feature_extractor.model_input_names[0]] = inputs.get(
            feature_extractor.model_input_names[0]
        )[0]
        batch["input_length"] = len(sample["array"])

        # process text
        # TODO: check manually the labels could be weird
        input_str = batch[data_args.text_column_name]
        input_str = text_normalizer.normalize(
            input_str, do_lower=data_args.do_lower_case
        )
        if data_args.do_remove_punctuation:
            input_str = text_normalizer.normalizer(input_str).strip()

        batch["labels"] = tokenizer(input_str).input_ids
        return batch

    def prepare_dataset_for_lora(batch):
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # encode target text to label ids
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        batch["input_length"] = len(audio["array"])
        return batch

    # process datasets
    prep_fn = (
        prepare_dataset_for_lora
        if hasattr(model_args, "use_lora") and model_args.use_lora
        else prepare_dataset
    )
    vectorized_datasets = raw_datasets.map(
        prep_fn,
        remove_columns=next(iter(raw_datasets.values())).features.keys(),
    ).with_format("torch")

    if training_args.do_train:
        vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(
            # buffer_size=data_args.shuffle_buffer_size,
            seed=training_args.seed,
        )

    # filter datasets based on audio length
    max_input_length = (
        data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    )
    min_input_length = (
        data_args.min_duration_in_seconds * feature_extractor.sampling_rate
    )

    def is_audio_in_length_range(length: int) -> bool:
        """Utility function for filtering training data.
        Checks if the input length is within the user-defined minimum and maximum range.
        """
        return min_input_length < length < max_input_length

    vectorized_datasets["train"] = vectorized_datasets["train"].filter(
        is_audio_in_length_range,
        input_columns=["input_length"],
    )

    # init metrics calculator
    metrics_calculator = MetricsCalculator(
        tokenizer=tokenizer, do_normalize_eval=data_args.do_normalize_eval
    )

    # save processor components
    if is_main_process(training_args.local_rank):
        feature_extractor.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        config.save_pretrained(training_args.output_dir)

    if hasattr(model_args, "use_lora") and model_args.use_lora:
        logger.info("Applying LoRA to the model")
        model = prepare_model_for_kbit_training(model)

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
        lora_config = lora_args.create_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # add some params in training args
        training_args.remove_unused_columns = False
        training_args.label_names = ["labels"]

    if model_args.freeze_feature_encoder:
        logger.info("Freezing feature encoder...")
        model.freeze_feature_encoder()
    if model_args.freeze_encoder:
        logger.info("Freezing encoder...")
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False

    # init trainer
    # TODO: add regularization
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=(
            metrics_calculator.compute_metrics
            if training_args.predict_with_generate
            else None
        ),
        callbacks=(
            [ShuffleCallback(), SavePeftModelCallback(save_total_limit=training_args.save_total_limit)]
            if model_args.use_lora
            else [ShuffleCallback()]
        ),
    )

    # init wandb
    wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        name=data_args.wandb_run_name,
        config={
            "model_name": model_args.model_name_or_path,
            "train_dataset_fraction": data_args.train_dataset_fraction,
            "eval_dataset_fraction": data_args.eval_dataset_fraction,
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "max_steps": training_args.max_steps,
            "dataset_config_path": data_args.dataset_config_path,
            "weight_decay": training_args.weight_decay,
            "dropout": model.config.dropout,
            "attention_dropout": model.config.attention_dropout,
            "activation_dropout": model.config.activation_dropout,
        },
    )

    # training
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
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

        logger.info(f"Training model from checkpoint {checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        # log and save metrics
        metrics = train_result.metrics
        if data_args.max_train_samples:
            metrics["train_samples"] = data_args.max_train_samples

        # Save the training log history as a CSV file
        df = pd.DataFrame(trainer.state.log_history)
        save_path = os.path.join(training_args.output_dir, "train_history.csv")
        df.to_csv(save_path, index=False)

        # log metrics using trainer's logger
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # add training metrics to cache
        mt.add_metrics(metrics, "train")

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

        # add eval metrics to cache
        mt.add_metrics(metrics, "eval")

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "automatic-speech-recognition",
        "tags": "whisper-event",
    }

    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset"] = (
                f"{data_args.dataset_name} {data_args.dataset_config_name}"
            )
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
