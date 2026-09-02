from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from webapp.inference import PROJECT_ROOT, decode_image


DEFAULT_HARD_CASE_DIR = PROJECT_ROOT / "dataset" / "hard_cases"
VALID_ISSUES = {"missed", "wrong-class", "false-positive", "difficult-condition"}
VALID_SOURCES = {"upload", "camera"}


class HardCaseStore:
    def __init__(self, root: Path | str = DEFAULT_HARD_CASE_DIR) -> None:
        self.root = Path(root).resolve()
        self.image_dir = self.root / "images"
        self.manifest_path = self.root / "manifest.jsonl"
        self._lock = threading.Lock()

    def save(
        self,
        payload: bytes,
        *,
        suffix: str,
        source: str,
        issue_type: str,
        expected_class: str | None,
        predicted_classes: list[str],
        notes: str,
        confidence_threshold: float,
    ) -> dict[str, Any]:
        if source not in VALID_SOURCES:
            raise ValueError("Source must be upload or camera.")
        if issue_type not in VALID_ISSUES:
            raise ValueError("Choose a valid issue type.")
        decode_image(payload)

        now = datetime.now(timezone.utc)
        record_id = f"{now.strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}"
        safe_suffix = suffix if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} else ".jpg"
        file_name = f"{record_id}{safe_suffix}"
        relative_image = Path("images") / file_name
        record = {
            "id": record_id,
            "created_at": now.isoformat(),
            "image": relative_image.as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "source": source,
            "issue_type": issue_type,
            "expected_class": expected_class or None,
            "predicted_classes": predicted_classes,
            "confidence_threshold": round(confidence_threshold, 2),
            "notes": notes.strip()[:300],
            "annotation_status": "unannotated",
        }

        with self._lock:
            self.image_dir.mkdir(parents=True, exist_ok=True)
            (self.image_dir / file_name).write_bytes(payload)
            with self.manifest_path.open("a", encoding="utf-8", newline="\n") as manifest:
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def count(self) -> int:
        if not self.manifest_path.is_file():
            return 0
        with self._lock:
            return sum(1 for line in self.manifest_path.read_text(encoding="utf-8").splitlines() if line.strip())

