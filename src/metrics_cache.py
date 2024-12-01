import json
import os
from typing import Dict, Any, List
from datetime import datetime


# TODO: this is kind of a hack, but it works for now
# need to find a way to secure the metrics from HF?
# need this for plotting later in the report
# other way si to parse from tensorboard?
class MetricsCache:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.train_metrics = {
            "train_runtime": [],
            "train_samples_per_second": [],
            "train_steps_per_second": [],
            "total_flos": [],
            "train_loss": [],
            "epoch": [],
        }
        self.eval_metrics = {
            "eval_cer": [],
            "eval_wer": [],
            "eval_loss": [],
            "eval_runtime": [],
            "eval_samples_per_second": [],
            "eval_steps_per_second": [],
            "epoch": [],
        }
        self.train_file = os.path.join(output_dir, "train_metrics_history.json")
        self.eval_file = os.path.join(output_dir, "eval_metrics_history.json")
        self._load_existing_cache()

    def _load_existing_cache(self):
        if os.path.exists(self.train_file):
            with open(self.train_file, "r") as f:
                self.train_metrics = json.load(f)
        if os.path.exists(self.eval_file):
            with open(self.eval_file, "r") as f:
                self.eval_metrics = json.load(f)

    def add_metrics(self, metrics: Dict[str, Any], prefix: str = "train"):
        metrics_dict = self.train_metrics if prefix == "train" else self.eval_metrics

        # timestamp for sorting?
        current_time = datetime.now().isoformat()

        for key, value in metrics.items():
            if key in metrics_dict:
                metrics_dict[key].append(value)

        self._save_cache(prefix)

    def _save_cache(self, prefix: str):
        os.makedirs(self.output_dir, exist_ok=True)
        file_path = self.train_file if prefix == "train" else self.eval_file
        metrics_dict = self.train_metrics if prefix == "train" else self.eval_metrics

        with open(file_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)

    def get_history(self, prefix: str = "train") -> Dict[str, List[float]]:
        return self.train_metrics if prefix == "train" else self.eval_metrics
