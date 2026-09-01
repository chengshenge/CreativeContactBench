"""Versioned prompt loading, hashing, and placeholder rendering."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


OPTION_IDS = ("A", "B", "C", "D")
PLACEHOLDERS = (
    "{{TASK_INSTRUCTION}}",
    "{{OPTION_A}}",
    "{{OPTION_B}}",
    "{{OPTION_C}}",
    "{{OPTION_D}}",
)
UNRESOLVED = re.compile(r"\{\{[^{}]+\}\}")


class PromptError(ValueError):
    """The benchmark prompt is missing or cannot be rendered exactly."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_prompt(template: str, instruction: str, options: dict[str, str]) -> str:
    if set(options) != set(OPTION_IDS):
        raise PromptError("Options must be exactly A, B, C, and D")
    replacements = {
        "{{TASK_INSTRUCTION}}": instruction,
        **{f"{{{{OPTION_{option}}}}}": options[option] for option in OPTION_IDS},
    }
    rendered = template
    for placeholder in PLACEHOLDERS:
        if rendered.count(placeholder) != 1:
            raise PromptError(f"Prompt must contain exactly one {placeholder}")
        rendered = rendered.replace(placeholder, replacements[placeholder])
    unresolved = UNRESOLVED.findall(rendered)
    if unresolved:
        raise PromptError(f"Unresolved prompt placeholders: {unresolved}")
    return rendered
