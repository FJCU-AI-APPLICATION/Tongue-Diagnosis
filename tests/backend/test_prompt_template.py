import pytest

from backend.llm import prompt


def test_validate_accepts_template_with_exactly_one_placeholder():
    prompt.validate("rules\n{{PREDICTIONS}}\nformat")  # no exception


def test_validate_rejects_missing_placeholder():
    with pytest.raises(prompt.PromptValidationError) as exc:
        prompt.validate("rules without marker")
    assert "{{PREDICTIONS}}" in str(exc.value)


def test_validate_rejects_duplicate_placeholder():
    with pytest.raises(prompt.PromptValidationError) as exc:
        prompt.validate("a {{PREDICTIONS}} b {{PREDICTIONS}} c")
    assert "2" in str(exc.value)


def test_render_substitutes_block_in_place():
    out = prompt.render(
        "before\n{{PREDICTIONS}}\nafter",
        "- 舌色：淡紅（0.78）",
    )
    assert out == "before\n- 舌色：淡紅（0.78）\nafter"


def test_render_validates_before_substitution():
    with pytest.raises(prompt.PromptValidationError):
        prompt.render("template with no marker", "- bullet")


def test_render_handles_empty_block_sentinel():
    out = prompt.render("a\n{{PREDICTIONS}}\nb", "- （無可用判讀資料）")
    assert "（無可用判讀資料）" in out
