import yaml
from typing import List, Optional
from datasets import (
    Audio,
    IterableDataset,
    interleave_datasets,
    load_dataset,
    concatenate_datasets,
)


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


def load_datasets_from_config(
    config_path: str,
    split: str,
    sampling_rate: Optional[int] = 16000,
    dataset_fraction: float = 1.0,
) -> IterableDataset:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    datasets_config = config["dataset_config"][split]["datasets"]
    all_datasets = []

    for dataset_config in datasets_config:
        dataset = load_dataset(
            dataset_config["name"],
            dataset_config["config"],
            trust_remote_code=True,
        )

        # Combine the pre-split datasets into one so we can make our own custom split
        if isinstance(dataset, dict):
            combined_dataset = concatenate_datasets([dataset[key] for key in dataset.keys()])
            dataset = combined_dataset

        # apply dataset fraction if less than 1
        if dataset_fraction < 1.0:
            num_examples = len(dataset)
            num_keep = int(num_examples * dataset_fraction)
            dataset = dataset.shuffle(seed=42).select(
                range(num_keep)
            )  # TODO: make this seed refer to the seed argument in args.py

        dataset = dataset.cast_column("audio", Audio(sampling_rate))

        if dataset_config["text_column"] != "sentence":
            dataset = dataset.rename_column(dataset_config["text_column"], "sentence")

        dataset = dataset.remove_columns(
            set(dataset.features.keys()) - set(["audio", "sentence"])
        )

        num_total_examples = len(dataset)
        dataset = dataset.shuffle(seed=42)

        # Create an 80:10:10 split for train, val, and test
        dataset_train_split = dataset.train_test_split(
            test_size=0.2, shuffle=False
        )
        dataset_val_test_split = dataset_train_split["test"].train_test_split(
            test_size=0.5, shuffle=False
        )
        if dataset_config["split"] == "train":
            dataset = dataset_train_split["train"]
        elif dataset_config["split"] == "val":
            dataset = dataset_val_test_split["train"]
        elif dataset_config["split"] == "test":
            dataset = dataset_val_test_split["test"]

        all_datasets.append(dataset)

    return concatenate_datasets(all_datasets)
