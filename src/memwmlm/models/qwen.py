"""Qwen-specific prompt preparation, isolated from generic scoring code."""

from __future__ import annotations


class QwenChatPromptFormatter:
    """Render task text as a Qwen user turn with an explicit thinking mode (disabled by default)."""

    def __init__(self, tokenizer: object, enable_thinking: bool = False) -> None:
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking

    def __call__(self, task_text: str) -> str:
        if not getattr(self.tokenizer, "chat_template", None):
            return task_text
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": task_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
