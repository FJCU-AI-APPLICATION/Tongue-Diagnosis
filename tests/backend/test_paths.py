from backend.stores.paths import (
    PROMPT_DEFAULT,
    PROMPT_CURRENT,
    LLM_DEFAULT,
    LLM_CURRENT,
    REGISTRY_DEFAULT,
    REGISTRY_CURRENT,
)


def test_paths_point_to_real_default_files():
    assert PROMPT_DEFAULT.exists(), PROMPT_DEFAULT
    assert LLM_DEFAULT.exists(), LLM_DEFAULT
    assert REGISTRY_DEFAULT.exists(), REGISTRY_DEFAULT


def test_current_paths_are_siblings_of_defaults():
    assert PROMPT_CURRENT.parent == PROMPT_DEFAULT.parent
    assert LLM_CURRENT.parent == LLM_DEFAULT.parent
    assert REGISTRY_CURRENT.parent == REGISTRY_DEFAULT.parent
