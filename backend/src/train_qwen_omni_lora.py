from __future__ import annotations

import argparse
import inspect
import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from src.qwen_omni_dataset import load_jsonl, materialize_messages


def parse_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


class JsonlConversationDataset(Dataset):
    def __init__(self, path: str):
        self.data = load_jsonl(path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.data[index]


class OmniChatCollator:
    def __init__(
        self,
        processor: Qwen2_5OmniProcessor,
        max_length: int,
        padding: bool = True,
    ) -> None:
        self.processor = processor
        self.max_length = max_length
        self.padding = padding
        self.audio_sampling_rate = getattr(processor.feature_extractor, "sampling_rate", 16000)

    def _tokenize_messages(
        self,
        messages: List[List[Dict[str, Any]]],
        add_generation_prompt: bool,
    ) -> Dict[str, torch.Tensor]:
        batch = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
        )
        return dict(batch)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        full_messages = [
            materialize_messages(sample["messages"], audio_sampling_rate=self.audio_sampling_rate)
            for sample in batch
        ]
        prompt_messages = [
            materialize_messages(sample["messages"][:-1], audio_sampling_rate=self.audio_sampling_rate)
            for sample in batch
        ]

        full_inputs = self._tokenize_messages(full_messages, add_generation_prompt=False)
        prompt_inputs = self._tokenize_messages(prompt_messages, add_generation_prompt=True)

        labels = full_inputs["input_ids"].clone()
        labels[full_inputs["attention_mask"] == 0] = -100

        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        for row_idx, prompt_len in enumerate(prompt_lengths):
            labels[row_idx, :prompt_len] = -100

        full_inputs["labels"] = labels
        return full_inputs


def build_model(
    model_name: str,
    torch_dtype: torch.dtype,
    load_in_4bit: bool,
):
    quantization_config = None
    model_kwargs = {
        "torch_dtype": torch_dtype,
    }

    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = quantization_config

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_name,
        **model_kwargs,
    )
    return model, quantization_config


def apply_lora(
    model,
    target_modules: List[str],
    rank: int,
    alpha: int,
    dropout: float,
    resume_from_adapter: str,
    gradient_checkpointing: bool,
):
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if resume_from_adapter:
        return PeftModel.from_pretrained(model, resume_from_adapter, is_trainable=True)

    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    return get_peft_model(model, lora_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen2.5-Omni Thinker on image+audio+text chats.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--val_jsonl", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume_from_adapter", type=str, default="")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target module names.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch_dtype = parse_torch_dtype(args.torch_dtype)

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_name)
    model, _ = build_model(args.model_name, torch_dtype=torch_dtype, load_in_4bit=args.load_in_4bit)
    model = apply_lora(
        model,
        target_modules=[module.strip() for module in args.target_modules.split(",") if module.strip()],
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        resume_from_adapter=args.resume_from_adapter,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    processor.tokenizer.padding_side = "right"

    train_dataset = JsonlConversationDataset(args.train_jsonl)
    eval_dataset = JsonlConversationDataset(args.val_jsonl) if args.val_jsonl else None
    collator = OmniChatCollator(processor=processor, max_length=args.max_length)

    training_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
        "remove_unused_columns": False,
        "bf16": (torch_dtype == torch.bfloat16),
        "fp16": (torch_dtype == torch.float16),
        "report_to": [],
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in signature.parameters:
        training_kwargs["evaluation_strategy"] = "epoch" if eval_dataset is not None else "no"
    elif "eval_strategy" in signature.parameters:
        training_kwargs["eval_strategy"] = "epoch" if eval_dataset is not None else "no"

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
