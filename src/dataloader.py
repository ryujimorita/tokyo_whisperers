import yaml
from typing import List, Optional
from datasets import Audio, IterableDataset, interleave_datasets, load_dataset, concatenate_datasets

def load_streaming_dataset(dataset_name: str, dataset_config_name: str, split: str = "train", **kwargs) -> IterableDataset:
    if "+" in split:
        dataset_splits = [
            load_dataset(dataset_name, dataset_config_name, split=split_name, streaming=True, **kwargs)
            for split_name in split.split("+")
        ]
        return interleave_datasets(dataset_splits)
    else:
        return load_dataset(dataset_name, dataset_config_name, split=split, streaming=True, **kwargs)

def load_datasets_from_config(config_path: str, split: str, sampling_rate: Optional[int] = 16000) -> IterableDataset:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    datasets_config = config['dataset_config'][split]['datasets']
    all_datasets = []
    
    for dataset_config in datasets_config:
        print(dataset_config['name'])
        dataset = load_dataset(
            dataset_config['name'], 
            dataset_config['config'], 
            split=dataset_config['split'],
            trust_remote_code=True,
            # streaming=False, # streaming not working? F
        )
        dataset = dataset.cast_column("audio", Audio(sampling_rate))
        
        if dataset_config['text_column'] != "sentence":
            dataset = dataset.rename_column(dataset_config['text_column'], "sentence")
        
        dataset = dataset.remove_columns(
            set(dataset.features.keys()) - set(["audio", "sentence"])
        )
        all_datasets.append(dataset)

    # TODO: could be trimmed datasets based on smallest length
    # return interleave_datasets(all_datasets)
    return concatenate_datasets(all_datasets)