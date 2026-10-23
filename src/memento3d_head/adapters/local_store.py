"""Local metadata storage."""
import json
from pathlib import Path


class LocalStore:
    """Write each report via a temporary sibling file."""
    def write_json(self, path: Path, value: dict) -> None:
        """Atomically replace metadata inside an exclusively created attempt."""
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
