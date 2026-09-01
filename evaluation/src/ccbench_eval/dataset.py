"""Read-only Hugging Face download and strict dataset validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from PIL import Image

from . import DATASET_REPO, DATASET_REVISION, EXPECTED_TASKS


FIELDS = frozenset(
    {
        "file_name",
        "task_id",
        "task_instruction",
        "option_A",
        "option_B",
        "option_C",
        "option_D",
    }
)
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"})
TASK_ID = re.compile(r"^task-(\d{2})$")


class DatasetValidationError(ValueError):
    """The downloaded snapshot does not match its declared contract."""


@dataclass(frozen=True)
class Task:
    file_name: str
    task_id: str
    task_instruction: str
    option_A: str
    option_B: str
    option_C: str
    option_D: str
    image_path: Path

    @property
    def options(self) -> dict[str, str]:
        return {"A": self.option_A, "B": self.option_B, "C": self.option_C, "D": self.option_D}


def download_dataset(
    repo_id: str = DATASET_REPO,
    revision: str = DATASET_REVISION,
    cache_dir: str | Path | None = None,
) -> Path:
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": revision,
        "allow_patterns": ["metadata.jsonl", "images/*"],
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return Path(snapshot_download(**kwargs))


def validate_local_dataset(root: str | Path, expected_tasks: int = EXPECTED_TASKS) -> list[Task]:
    root = Path(root)
    metadata = root / "metadata.jsonl"
    if not metadata.is_file():
        raise DatasetValidationError(f"Missing metadata file: {metadata}")
    expected_ids = {f"task-{number:02d}" for number in range(1, expected_tasks + 1)}
    records: list[Task] = []
    seen: set[str] = set()
    for line_number, line in enumerate(metadata.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(f"Invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(raw, dict) or set(raw) != FIELDS:
            raise DatasetValidationError(
                f"Metadata line {line_number} fields differ from the contract"
            )
        if any(not isinstance(raw[field], str) or not raw[field].strip() for field in FIELDS):
            raise DatasetValidationError(f"Metadata line {line_number} contains an empty field")
        task_id = raw["task_id"]
        if not TASK_ID.fullmatch(task_id) or task_id in seen:
            raise DatasetValidationError(f"Invalid or duplicate task ID: {task_id}")
        seen.add(task_id)
        relative_image = Path(raw["file_name"])
        if (
            relative_image.is_absolute()
            or ".." in relative_image.parts
            or not relative_image.parts
            or relative_image.parts[0] != "images"
        ):
            raise DatasetValidationError(
                f"Image path for {task_id} must remain under images/: {raw['file_name']}"
            )
        # Hugging Face snapshots use symlinks into their content-addressed blob
        # store, so enforce containment lexically before following the file.
        image_path = root / relative_image
        if not image_path.is_file():
            raise DatasetValidationError(f"Missing image for {task_id}: {raw['file_name']}")
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            raise DatasetValidationError(f"Unreadable image for {task_id}: {exc}") from exc
        records.append(Task(**{field: raw[field] for field in FIELDS}, image_path=image_path))
    if len(records) != expected_tasks or seen != expected_ids:
        raise DatasetValidationError(
            f"Expected {expected_tasks} continuous task IDs; found {len(records)} records, "
            f"missing={sorted(expected_ids - seen)}, unexpected={sorted(seen - expected_ids)}"
        )
    image_files = {
        path.resolve()
        for path in (root / "images").rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    referenced = {record.image_path.resolve() for record in records}
    if len(image_files) != expected_tasks or image_files != referenced:
        raise DatasetValidationError("Image inventory does not match metadata exactly")
    return sorted(records, key=lambda record: int(record.task_id.split("-")[1]))


def load_dataset(
    repo_id: str = DATASET_REPO,
    revision: str = DATASET_REVISION,
    expected_tasks: int = EXPECTED_TASKS,
    cache_dir: str | Path | None = None,
) -> tuple[Path, list[Task]]:
    root = download_dataset(repo_id, revision, cache_dir)
    return root, validate_local_dataset(root, expected_tasks)
