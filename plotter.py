import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from typing import Dict, List
import argparse
from loguru import logger

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def get_experiment_colors(experiments: List[dict], num_experiments: int) -> Dict[str, str]:
    default_colors = plt.cm.rainbow(np.linspace(0, 1, num_experiments))
    colors = {}
    for idx, exp in enumerate(experiments):
        if 'color' in exp:
            colors[exp['name']] = exp['color']
        else:
            colors[exp['name']] = default_colors[idx]
    return colors

def load_training_data(experiment_path: str) -> pd.DataFrame:
    metrics_path = Path(experiment_path) / "training_metrics.csv"
    df = pd.read_csv(metrics_path)
    df = df.groupby('epoch').mean(numeric_only=True)
    return df

def smooth_data(data: np.ndarray, window_size: int) -> np.ndarray:
    return uniform_filter1d(data, size=window_size)

def calculate_window_std(values: np.ndarray, window_size: int) -> np.ndarray:
    window_std = np.array([
        np.std(values[max(0, i-window_size//2):min(len(values), i+window_size//2)])
        for i in range(len(values))
    ])
    return window_std

def plot_experiment(ax, data: pd.DataFrame, metric: str, exp_name: str, 
                   color: str, window_size: int, show_std: bool = True):
    if metric not in data.columns:
        logger.warning(f"Metric {metric} not found in data for {exp_name}")
        return
        
    values = data[metric].dropna()
    if len(values) == 0:
        logger.warning(f"No valid data for metric {metric} in {exp_name}")
        return
        
    epochs = values.index.values
    values = values.values
        
    smoothed_values = smooth_data(values, window_size)
    
    ax.plot(epochs, smoothed_values, label=None, color="black", linewidth=5)
    ax.plot(epochs, smoothed_values, label=exp_name, color=color, linewidth=4)
    
    if show_std:
        window_std = calculate_window_std(values, window_size)
        smoothed_std = smooth_data(window_std, window_size)
        
        ax.fill_between(epochs,
                       smoothed_values - smoothed_std,
                       smoothed_values + smoothed_std,
                       color=color, alpha=0.3)

def create_plots(config: dict):
    experiments = config['experiments']
    colors = get_experiment_colors(experiments, len(experiments))
    window_size = config.get('window_size', 10)
    show_std = config.get('show_std', True)
    metrics_to_plot = config.get('metrics_to_plot', ['train_loss', 'eval_loss'])
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for exp in experiments:
            exp_name = exp['name']
            data = load_training_data(exp['path'])
            plot_experiment(ax, data, metric, exp_name, 
                          colors[exp_name], window_size, show_std)
        
        ax.set_xlabel("Epochs", fontsize=30)
        
        # Special handling for y-axis labels
        if 'cer' in metric.lower():
            parts = metric.replace('_', ' ').title().split()
            parts = [p.upper() if p.lower() == 'cer' else p for p in parts]
            ylabel = ' '.join(parts)
        elif 'wer' in metric.lower():
            parts = metric.replace('_', ' ').title().split()
            parts = [p.upper() if p.lower() == 'wer' else p for p in parts]
            ylabel = ' '.join(parts)
        elif 'loss' in metric.lower():
            if metric.startswith('train_'):
                ylabel = 'Train Loss'
            elif metric.startswith('eval_'):
                ylabel = 'Eval Loss'
            else:
                ylabel = 'Loss'
        else:
            ylabel = metric.replace('_', ' ').title()
            
        ax.set_ylabel(ylabel, fontsize=30)
        ax.legend(fontsize=22)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.tick_params(axis="both", which="major", labelsize=22)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 2))
        plt.tight_layout()
        
        save_path = output_dir / f"{metric}_line.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from experiments")
    parser.add_argument("config", type=str, help="Path to the configuration YAML file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    create_plots(config)

if __name__ == "__main__":
    main() 