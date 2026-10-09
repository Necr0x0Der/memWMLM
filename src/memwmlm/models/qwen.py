"""Qwen-specific prompt preparation, isolated from generic scoring code."""

from __future__ import annotations


class QwenChatPromptFormatter:
    """Render task text as a Qwen user turn with thinking explicitly disabled."""

    def __init__(self, tokenizer: object) -> None:
        self.tokenizer = tokenizer

    def __call__(self, task_text: str) -> str:
        if not getattr(self.tokenizer, "chat_template", None):
            return task_text
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": task_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
