import os
import yaml
import math
from typing import Optional, Tuple, Dict
from datasets import (
    Audio,
    Dataset,
    IterableDataset,
    interleave_datasets,
    load_dataset,
    concatenate_datasets,
)
from loguru import logger


def load_streaming_dataset(
    dataset_name: str, dataset_config_name: str, split: str = "train", **kwargs
) -> IterableDataset:
    if "+" in split:
        dataset_splits = [
            load_dataset(
                dataset_name,
                dataset_config_name,
                split=split_name,
                streaming=True,
                **kwargs
            )
            for split_name in split.split("+")
        ]
        return interleave_datasets(dataset_splits)
    else:
        return load_dataset(
            dataset_name, dataset_config_name, split=split, streaming=True, **kwargs
        )

def train_val_test_split(
        dataset, 
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    ) -> Dict[str, Dataset]:
    """
    Utility function to perform a train-val-test split. 
    Defaults to an 80:10:10 split but can be configured if needed.
    """
    # Check if the split ratio adds up to 1
    assert math.isclose(sum(split_ratio), 1, rel_tol=1e-9), "Split_ratio doesn't add up to 100%. Check the values."

    train_size, val_size, test_size = split_ratio

    # Split dataset into train data and test data
    dataset_train_test_split = dataset.train_test_split(
        test_size=val_size + test_size, 
        shuffle=False
    )

    # Split the test data into val data and test data
    dataset_val_test_split = dataset_train_test_split["test"].train_test_split(
        test_size=test_size / (val_size + test_size),  # => 0.5 for an 80:10:10 split
        shuffle=False
    )

    res = {
        "train": dataset_train_test_split["train"],
        "eval": dataset_val_test_split["train"],
        "test": dataset_val_test_split["test"]
    }

    # Log the sizes of each split
    logger.info(f"Dataset split completed. Train size: {len(res['train'])}, "
                f"Eval size: {len(res['eval'])}, Test size: {len(res['test'])}")

    return res

def load_datasets_from_config(
    config_path: str,
    data_type: str,
    sampling_rate: Optional[int] = 16000,
    dataset_fraction: float = 1.0,
) -> Dataset:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    all_datasets_config = config["dataset_config"][data_type]["datasets"]
    all_datasets = []

    logger.info(f"Loading {len(all_datasets_config)} datasets")
    for dataset_config in all_datasets_config:
        try:
            dataset = load_dataset(
                dataset_config["name"],
                dataset_config["config"],
                split=dataset_config["split"],
                trust_remote_code=True,
            )
        except ValueError as e:
            # Some datasets, like ReazonSpeech may need an access token. Please set your access token in the HF_TOKEN environment variable.
            try:
                access_token = os.environ.get('HF_TOKEN')
                dataset = load_dataset(
                    dataset_config["name"],
                    dataset_config["config"],
                    split=dataset_config["split"],
                    trust_remote_code=True,
                    token=access_token,
                )
            except:
                logger.error(f"Failed to load dataset {dataset_config['name']}")
                logger.error(f"Did you set the HF_TOKEN environment variable?")
                raise
        except:
            logger.error(f"Failed to load dataset {dataset_config['name']}")
            raise

        # If split is needed, apply the split, retrieve the dataset matching the data type, and reinsert it into the dataset variable for merging later
        if dataset_config["need_additional_split"]:
            if len(dataset) > 0:
                logger.info(f"Additional split is required because {dataset_config['name']} doesn't have train, validation, test split metadata in huggingface dataset.")
                dataset = train_val_test_split(dataset)[data_type]
            else:
                logger.warning(f"Dataset {dataset_config['name']} is empty. Skipping additional split.")

        logger.info(f"Loaded {len(dataset)} examples from {dataset_config['name']}")
        dataset = dataset.cast_column("audio", Audio(sampling_rate))

        if dataset_config["text_column"] != "sentence":
            dataset = dataset.rename_column(dataset_config["text_column"], "sentence")

        dataset = dataset.remove_columns(
            set(dataset.features.keys()) - set(["audio", "sentence"])
        )

        # shuffle the dataset
        dataset = dataset.shuffle(seed=42)

        # apply dataset fraction if less than 1
        if dataset_fraction < 1.0:
            num_examples = len(dataset)
            num_keep = int(num_examples * dataset_fraction)
            dataset = dataset.select(
                range(num_keep)
            )  # TODO: make this seed refer to the seed argument in args.py
            logger.info(f"Applying dataset fraction of {dataset_fraction} resulting in {num_keep} examples")

        all_datasets.append(dataset)

    # log how many total examples we have
    total_examples = sum([len(ds) for ds in all_datasets])
    logger.info(f"Concatenating all datasets to create the {data_type} split using {len(all_datasets)} datasets containing a total of {total_examples} examples")

    return concatenate_datasets(all_datasets)