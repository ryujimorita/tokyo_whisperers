from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from tqdm.auto import tqdm
from torch.utils.data import IterableDataset
import os


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

    def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
        if isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(train_dataloader.dataset._epoch + 1)
            
class SavePeftModelCallback(TrainerCallback):
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
        return control
