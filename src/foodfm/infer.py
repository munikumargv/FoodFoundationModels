"""Load a base model (+ optional FoodFM LoRA adapter) and answer grounded questions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .chat_format import normalize, render, supports_system_role
from .grounding import Grounder
from .knowledge import load_kb
from .profile import Profile


class FoodFM:
    def __init__(self, model_key: str | None = None, adapter_dir: str | None = None, load_in_4bit: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.kb = load_kb()
        self.grounder = Grounder(self.kb)

        meta_path = Path(adapter_dir) / "foodfm_meta.json" if adapter_dir else None
        if meta_path and meta_path.exists() and not model_key:
            hf_id = json.loads(meta_path.read_text())["base_model"]
        else:
            hf_id = self.kb.model(model_key)["hf_id"]
        self.hf_id = hf_id

        use_cuda = torch.cuda.is_available()
        quant = None
        if load_in_4bit and use_cuda:
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir or hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, quantization_config=quant, device_map="auto" if use_cuda else None,
            torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        )
        if adapter_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_dir)
        self.model = model.eval()
        self.system_ok = supports_system_role(self.tokenizer)

    def generate(self, messages: list[dict[str, str]], max_new_tokens: int = 700, temperature: float = 0.3) -> str:
        import torch

        text = render(self.tokenizer, normalize(messages, self.system_ok), add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None, top_p=0.9 if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def ask(self, question: str, profile: Profile | None = None, **gen: Any) -> str:
        return self.generate(self.grounder.messages(question, profile=profile), **gen)
