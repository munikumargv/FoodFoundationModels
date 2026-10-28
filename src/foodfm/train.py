"""QLoRA fine-tuning of an open LLM on the FoodFM nutrition dataset.

Requires the optional training dependencies: pip install -e ".[train]"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .chat_format import supports_system_role, tokenize_example
from .dataset import read_jsonl
from .knowledge import CONFIG_DIR, load_kb


def load_train_config(path: str | Path | None = None, **overrides: Any) -> dict[str, Any]:
    with open(path or CONFIG_DIR / "train.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


class _Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(b["input_ids"]) for b in batch)

        def pad(key: str, value: int) -> Any:
            return torch.tensor([b[key] + [value] * (width - len(b[key])) for b in batch])

        return {"input_ids": pad("input_ids", self.pad_id), "attention_mask": pad("attention_mask", 0), "labels": pad("labels", -100)}


def train(model_key: str | None = None, config_path: str | None = None, **overrides: Any) -> str:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    cfg = load_train_config(config_path, **overrides)
    model_info = load_kb().model(model_key)
    hf_id = model_info["hf_id"]
    print(f"Base model: {hf_id}  ({model_info.get('license')})")

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    system_ok = supports_system_role(tokenizer)

    def encode(path: str) -> list[dict[str, list[int]]]:
        rows = [tokenize_example(tokenizer, ex["messages"], cfg["max_seq_length"], system_ok) for ex in read_jsonl(path)]
        return [r for r in rows if any(label != -100 for label in r["labels"])]

    train_rows = encode(cfg["dataset_path"])
    eval_rows = encode(cfg["eval_path"]) if cfg.get("eval_path") and Path(cfg["eval_path"]).exists() else None
    print(f"Train examples: {len(train_rows)}; eval examples: {len(eval_rows or [])}")

    use_cuda = torch.cuda.is_available()
    quant = None
    if cfg["load_in_4bit"] and use_cuda:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else (torch.float16 if use_cuda else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(hf_id, quantization_config=quant, torch_dtype=dtype, device_map="auto" if use_cuda else None)
    model.config.use_cache = False
    if quant is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    out_dir = cfg["output_dir"]
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        eval_strategy="epoch" if eval_rows else "no",
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=use_cuda,
        report_to="none",
        seed=cfg["seed"],
        max_steps=cfg.get("max_steps", -1),
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_rows, eval_dataset=eval_rows, data_collator=_Collator(tokenizer.pad_token_id))
    trainer.train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(Path(out_dir) / "foodfm_meta.json", "w", encoding="utf-8") as f:
        json.dump({"base_model": hf_id, "model_key": model_info["key"], "train_config": cfg}, f, indent=2)
    print(f"LoRA adapter saved to {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="QLoRA fine-tune an open LLM on FoodFM data")
    ap.add_argument("--model", help="key from configs/models.yaml or a Hugging Face id")
    ap.add_argument("--config", help="training config YAML (default configs/train.yaml)")
    ap.add_argument("--output-dir")
    ap.add_argument("--epochs", type=float, dest="num_train_epochs")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--no-4bit", action="store_true")
    a = ap.parse_args(argv)
    train(a.model, a.config, output_dir=a.output_dir, num_train_epochs=a.num_train_epochs, max_steps=a.max_steps,
          load_in_4bit=False if a.no_4bit else None)


if __name__ == "__main__":
    main()
