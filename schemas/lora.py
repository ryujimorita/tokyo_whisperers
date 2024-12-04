from dataclasses import dataclass, field
from typing import Optional
from peft import LoraConfig

@dataclass
class LoRAArguments:
    """Arguments for LoRA fine-tuning"""
    lora_r: int = field(
        default=8,
        metadata={"help": "Rank of the LoRA update matrices"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "Scaling factor for LoRA"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for LoRA layers"}
    )
    lora_target_modules: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={"help": "Comma-separated list of target modules to apply LoRA"}
    )

    def create_config(self):
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules.split(","),
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )
