"""System prompt template — validate and render the {{PREDICTIONS}} placeholder."""

from __future__ import annotations


PLACEHOLDER = "{{PREDICTIONS}}"


class PromptValidationError(ValueError):
    """Raised when a prompt template does not contain exactly one {{PREDICTIONS}}."""


def validate(template: str) -> None:
    count = template.count(PLACEHOLDER)
    if count == 0:
        raise PromptValidationError(
            f"提示詞必須包含 {PLACEHOLDER} 標記，目前缺少。"
        )
    if count > 1:
        raise PromptValidationError(
            f"提示詞只能有一個 {PLACEHOLDER} 標記，目前有 {count} 個。"
        )


def render(template: str, predictions_block: str) -> str:
    validate(template)
    return template.replace(PLACEHOLDER, predictions_block)
