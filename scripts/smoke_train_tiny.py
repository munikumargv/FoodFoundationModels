"""Offline end-to-end smoke test of the training and inference code paths.

Builds a tiny randomly initialised Llama-architecture model and a BPE tokenizer
(with a ChatML chat template) from the FoodFM dataset itself, so no model
download is needed. It checks that dataset -> LoRA training -> adapter loading
-> grounded generation all run; the output text is meaningless.

    python scripts/smoke_train_tiny.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CHATML = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def build_tiny_model(path: Path, texts: list[str]) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    specials = ["<pad>", "<|im_start|>", "<|im_end|>"]
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=2000, special_tokens=specials,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    from tokenizers import decoders

    tok.decoder = decoders.ByteLevel()
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<|im_end|>", bos_token="<|im_start|>")
    hf_tok.chat_template = CHATML
    hf_tok.save_pretrained(path)

    cfg = LlamaConfig(vocab_size=len(hf_tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096,
                      pad_token_id=hf_tok.pad_token_id, eos_token_id=hf_tok.eos_token_id, bos_token_id=hf_tok.bos_token_id)
    LlamaForCausalLM(cfg).save_pretrained(path)


def main() -> None:
    from foodfm.dataset import build_dataset, write_jsonl
    from foodfm.infer import FoodFM
    from foodfm.profile import Profile
    from foodfm.train import train

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        train_rows, eval_rows = build_dataset(n_profiles=40, seed=0)
        write_jsonl(train_rows, tmp_path / "train.jsonl")
        write_jsonl(eval_rows, tmp_path / "eval.jsonl")
        texts = [m["content"] for ex in train_rows for m in ex["messages"]]
        build_tiny_model(tmp_path / "tiny", texts)

        out = train(str(tmp_path / "tiny"), dataset_path=str(tmp_path / "train.jsonl"), eval_path=str(tmp_path / "eval.jsonl"),
                    output_dir=str(tmp_path / "lora"), max_steps=3, load_in_4bit=False, per_device_train_batch_size=2,
                    gradient_accumulation_steps=1, max_seq_length=1024, save_steps=1000)

        fm = FoodFM(adapter_dir=out, load_in_4bit=False)
        p = Profile(country="india", age=30, sex="female", weight_kg=55, height_cm=158)
        answer = fm.ask("Give me a personalised nutrition plan.", profile=p, max_new_tokens=16, temperature=0)
        print("Generated (random weights, meaningless):", repr(answer[:120]))
        print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
