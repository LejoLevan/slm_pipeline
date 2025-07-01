"""LoRA arguments class"""
from dataclasses import dataclass, field
from typing import Optional, List
from peft import LoraConfig, TaskType

@dataclass
class LoraArguments:
    """class contains all relevant parameters related to LoRA
    """

    if_lora: bool = field(
        default=False,
        metadata={"help": "whether or not LoRA is used in training"}
    )
    model_name_or_path: str = field(
        metadata={"help": "pretrained model system path or model identifier from huggingface"}
    )
    r: int = field(
        default=8,
        metadata={"help": "LoRA matrices rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "applied to LoRA output (output * alpha / r)"}
    )
    lora_dropout: int = field(
        default=0.01,
        metadata={"help": "applied before LoRA layers during training, prevents overfitting"}
    )
    target_modules: List[str] = field(
        default=None,
        metadata={"help": "list of submodule names where LoRA should be applied"}
    )
    bias: str = field(
        default="none",
        metadata={"help": "bias handling: 'none', 'all', or 'lora_only'"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "set to True for models with transposed weights (e.g., GPT-style)"}
    )
    task_type: str = field(
        default="QUESTION_ANSWERING",
        metadata={"help": "task type for PEFT"}
    )

    def guess_targets(self) -> Optional[List[str]]:
        name = self.model_name_or_path.lower()
        if "bert" in name or "roberta" in name:
            return ["query", "key", "value"]
        if "deberta" in name:
            return ["query_proj", "value_proj"]
        if "gpt2" in name:
            return ["c_attn"]
        if "llama" in name or "mistral" in name:
            return ["q_proj", "v_proj"]
        if "bloom" in name:
            return ["query_key_value"]
        if "t5" in name:
            return ["q", "v"]
        return None
    
    def config(self) -> Optional[LoraConfig]:
        targets = self.target_modules or self.guess_targets()
        if not targets:
            raise ValueError("--LoRA config requires target modules")
        
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=targets,
            bias=self.bias,
            fan_in_fan_out=self.fan_in_fan_out,
            task_type=TaskType[self.task_type],
        )


