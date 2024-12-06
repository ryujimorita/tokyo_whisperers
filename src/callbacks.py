from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from tqdm.auto import tqdm
from torch.utils.data import IterableDataset
import os
import shutil
import pandas as pd
from typing import Dict


class EpochProgressCallback(TrainerCallback):
    """Custom callback for displaying epoch progress with tqdm"""

    def __init__(self):
        self.current_epoch = 0
        self.epoch_pbar = None
        self.total_steps = None
        self.completed_steps = 0

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control, **kwargs
    ):
        """Called at the beginning of training"""
        if state.is_world_process_zero:
            self.total_steps = args.num_train_epochs * args.num_update_steps_per_epoch
            print(f"\nStarting training for {args.num_train_epochs} epochs")

    def on_epoch_begin(
        self, args: TrainingArguments, state: TrainerState, control, **kwargs
    ):
        """Called at the beginning of each epoch"""
        if state.is_world_process_zero:  # only show progress bar on main process
            self.current_epoch += 1
            desc = f"Epoch {self.current_epoch}/{args.num_train_epochs}"
            self.epoch_pbar = tqdm(
                total=args.num_update_steps_per_epoch, desc=desc, leave=True, position=0
            )
            self.completed_steps = (
                self.current_epoch - 1
            ) * args.num_update_steps_per_epoch

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control, **kwargs
    ):
        """Update progress bar after each step"""
        if state.is_world_process_zero and self.epoch_pbar is not None:
            self.epoch_pbar.update(1)
            # Update global progress
            global_progress = (
                (self.completed_steps + self.epoch_pbar.n) / self.total_steps * 100
            )
            self.epoch_pbar.set_postfix({"global_progress": f"{global_progress:.1f}%"})

    def on_epoch_end(
        self, args: TrainingArguments, state: TrainerState, control, **kwargs
    ):
        """Called at the end of each epoch"""
        if state.is_world_process_zero and self.epoch_pbar is not None:
            self.epoch_pbar.close()
            self.epoch_pbar = None

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control, **kwargs
    ):
        """Called at the end of training"""
        if state.is_world_process_zero:
            print("\nTraining completed!")


class ShuffleCallback(TrainerCallback):
    """Callback to handle dataset shuffling between epochs"""

    def on_epoch_begin(self, args, state, control, train_dataloader=None, **kwargs):
        if train_dataloader is None:
            return
            
        try:
            if isinstance(train_dataloader.dataset, IterableDataset):
                train_dataloader.dataset.set_epoch(state.epoch)
            elif hasattr(train_dataloader.dataset, 'shuffle'):
                train_dataloader.dataset.shuffle(seed=args.seed + state.epoch)
                
        except Exception as e:
            logger.warning(f"Failed to shuffle dataset: {str(e)}")
            
        return control


class SavePeftModelCallback(TrainerCallback):
    def __init__(self, save_total_limit: int = None):
        self.save_total_limit = save_total_limit
        self.best_metric = float('inf')  # for WER, lower is better

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs,
    ):
        # check if we have a new best model based on WER
        if metrics.get("eval_wer", float('inf')) < self.best_metric:
            self.best_metric = metrics["eval_wer"]
            
            # save the best model
            best_model_path = os.path.join(args.output_dir, "best_lora_model")
            if os.path.exists(best_model_path):
                shutil.rmtree(best_model_path)
            
            kwargs["model"].save_pretrained(best_model_path)
            
            # log the new best metric
            print(f"New best model saved with WER: {self.best_metric}")

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"lora-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

        # handle checkpoint limit if specified
        if self.save_total_limit is not None:
            checkpoints = [
                f for f in os.listdir(args.output_dir) 
                if f.startswith("lora-") and os.path.isdir(os.path.join(args.output_dir, f))
            ]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            
            if len(checkpoints) > self.save_total_limit:
                num_to_remove = len(checkpoints) - self.save_total_limit
                removing = checkpoints[:num_to_remove]
                for checkpoint in removing:
                    full_path = os.path.join(args.output_dir, checkpoint)
                    shutil.rmtree(full_path)
        
        return control


class SaveBestModelCallback(TrainerCallback):
    def __init__(self, save_total_limit: int = None):
        self.save_total_limit = save_total_limit
        self.best_metric = float('inf')  # for WER, lower is better

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs,
    ):
        # check if we have a new best model based on WER
        if metrics.get("eval_wer", float('inf')) < self.best_metric:
            self.best_metric = metrics["eval_wer"]
            
            # save the best model
            best_model_path = os.path.join(args.output_dir, "best_model")
            if os.path.exists(best_model_path):
                shutil.rmtree(best_model_path)
            
            kwargs["model"].save_pretrained(best_model_path)
            
            # save the tokenizer if it exists
            if "tokenizer" in kwargs:
                kwargs["tokenizer"].save_pretrained(best_model_path)
            
            print(f"New best model saved with WER: {self.best_metric}")

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.save_total_limit is not None:
            checkpoints = [
                f for f in os.listdir(args.output_dir) 
                if f.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, f))
            ]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            
            if len(checkpoints) > self.save_total_limit:
                num_to_remove = len(checkpoints) - self.save_total_limit
                removing = checkpoints[:num_to_remove]
                for checkpoint in removing:
                    full_path = os.path.join(args.output_dir, checkpoint)
                    shutil.rmtree(full_path)
        
        return control


class MetricsSavingCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.log_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs is not None:
            self.log_history.append(logs)
            # save to CSV after each log
            df = pd.DataFrame(self.log_history)
            save_path = os.path.join(self.output_dir, "training_metrics.csv")
            df.to_csv(save_path, index=False)
