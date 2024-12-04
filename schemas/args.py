from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    feature_extractor_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "feature extractor name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `huggingface-cli login`"
        },
    )
    freeze_feature_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the feature encoder layers of the model."},
    )
    freeze_encoder: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the entire encoder of the seq2seq model."},
    )
    forced_decoder_ids: List[List[int]] = field(
        default=None,
        metadata={
            "help": "A list of pairs of integers which indicates a mapping from generation indices to token indices"
        },
    )
    suppress_tokens: List[int] = field(
        default=None,
        metadata={"help": "A list of tokens that will be suppressed at generation."},
    )
    model_index_name: str = field(
        default=None, metadata={"help": "Pretty name for the model card."}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning"}
    )


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    wandb_run_name: str = field(
        default="whisper_finetune",
        metadata={"help": "Name of the training run, used for wandb logging"},
    )
    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use"}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the full texts"
        },
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "For debugging purposes or quicker training"}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "For debugging purposes or quicker training"}
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data"},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={"help": "Truncate audio files longer than max_duration_in_seconds"},
    )
    min_duration_in_seconds: float = field(
        default=0.0,
        metadata={"help": "Filter audio files shorter than min_duration_in_seconds"},
    )
    do_lower_case: bool = field(
        default=False,
        metadata={"help": "Whether the target text should be lower cased"},
    )
    do_remove_punctuation: bool = field(
        default=False, metadata={"help": "Whether to remove punctuation"}
    )
    do_normalize_eval: bool = field(
        default=True,
        metadata={"help": "Whether to normalize the references and predictions"},
    )
    language: str = field(
        default=None, metadata={"help": "Language for multilingual fine-tuning"}
    )
    task: str = field(
        default="transcribe",
        metadata={"help": "Task, either `transcribe` or `translate`"},
    )
    shuffle_buffer_size: Optional[int] = field(
        default=500,
        metadata={"help": "The number of examples to download before shuffling"},
    )
    dataset_config_path: str = field(
        default="conf/dataset_config.yaml",
        metadata={"help": "Path to the dataset configuration YAML file"},
    )
    train_dataset_fraction: float = field(
        default=1.0,
        metadata={"help": "Fraction of training dataset to use (between 0 and 1)"},
    )
    eval_dataset_fraction: float = field(
        default=1.0,
        metadata={"help": "Fraction of evaluation dataset to use (between 0 and 1)"},
    )
    do_augment: bool = field(
        default=False,
        metadata={
            "help": "Apply data augmentation to the training set if specified"
        },
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for all fully connected layers"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for activation layers"},
    )
    # weight_decay: float = field(
    #     default=0.01,
    #     metadata={"help": "Weight decay for AdamW optimizer"},
    # )
