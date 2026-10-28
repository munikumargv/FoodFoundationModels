"""Chat-template helpers shared by training and inference."""

from __future__ import annotations

from typing import Any

Messages = list[dict[str, str]]


def supports_system_role(tokenizer: Any) -> bool:
    try:
        tokenizer.apply_chat_template([{"role": "system", "content": "x"}, {"role": "user", "content": "y"}], tokenize=False)
        return True
    except Exception:  # e.g. Gemma templates raise on a system role
        return False


def normalize(messages: Messages, system_ok: bool) -> Messages:
    """Merge the system prompt into the first user turn for templates without a system role."""
    if system_ok or not messages or messages[0]["role"] != "system":
        return messages
    system, rest = messages[0]["content"], list(messages[1:])
    if rest and rest[0]["role"] == "user":
        rest[0] = {"role": "user", "content": f"{system}\n\n{rest[0]['content']}"}
    return rest


def render(tokenizer: Any, messages: Messages, add_generation_prompt: bool) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    # Qwen3 hybrid-thinking templates: keep answers direct (no <think> traces).
    if "enable_thinking" in (getattr(tokenizer, "chat_template", "") or ""):
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def tokenize_example(tokenizer: Any, messages: Messages, max_length: int, system_ok: bool) -> dict[str, list[int]]:
    """Tokenize a conversation, masking everything except the final assistant reply."""
    messages = normalize(messages, system_ok)
    prompt = render(tokenizer, messages[:-1], add_generation_prompt=True)
    full = render(tokenizer, messages, add_generation_prompt=False)
    # Some templates render the prompt slightly differently from the prefix of
    # the full conversation; mask up to the longest common prefix.
    n = 0
    for a, b in zip(prompt, full):
        if a != b:
            break
        n += 1
    prompt_ids = tokenizer(full[:n], add_special_tokens=False)["input_ids"]
    input_ids = tokenizer(full, add_special_tokens=False)["input_ids"][:max_length]
    labels = [-100] * min(len(prompt_ids), len(input_ids)) + input_ids[len(prompt_ids):]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels[: len(input_ids)]}
