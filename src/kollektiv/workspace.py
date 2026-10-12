"""Text artifact storage with confinement, atomic writes and deterministic checks."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath


def relative_path(value: str) -> str:
    value = value.replace("\\", "/")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or PureWindowsPath(value).drive
        or any(part in ("", ".", "..") for part in value.split("/"))
        or any(ch in value for ch in ':\x00<>"|?*')
        or any(part.rstrip(" .") != part for part in path.parts)
        or any(
            part.split(".")[0].upper()
            in {
                "CON",
                "PRN",
                "AUX",
                "NUL",
                *[f"COM{i}" for i in range(1, 10)],
                *[f"LPT{i}" for i in range(1, 10)],
            }
            for part in path.parts
        )
    ):
        raise ValueError(f"Expected a safe relative file path, got {value!r}")
    return path.as_posix()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(prefix=".kollektiv-", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class Workspace:
    def __init__(self, root: Path, max_bytes: int = 2_000_000):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes

    def path(self, name: str) -> Path:
        path = (self.root / relative_path(name)).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Path leaves the artifact workspace")
        return path

    def list(self) -> list[str]:
        return sorted(
            p.relative_to(self.root).as_posix()
            for p in self.root.rglob("*")
            if p.is_file()
            and p.resolve().is_relative_to(self.root)
            and not p.name.startswith(".kollektiv-")
        )

    def read(self, name: str) -> str:
        path = self.path(name)
        if path.stat().st_size > self.max_bytes:
            raise ValueError(f"File exceeds {self.max_bytes} bytes")
        return path.read_text(encoding="utf-8")

    def write(self, name: str, content: str) -> dict:
        if not content.strip():
            raise ValueError("Deliverables must contain content")
        if len(content.encode("utf-8")) > self.max_bytes:
            raise ValueError(f"File exceeds {self.max_bytes} bytes")
        atomic_write(self.path(name), content)
        return self.inspect(name)

    def inspect(self, name: str) -> dict:
        content = self.read(name)
        problems = []
        if not content.strip():
            problems.append("File is empty")
        suffix = Path(name).suffix.lower()
        try:
            if suffix == ".json":
                json.loads(content)
            elif suffix == ".py":
                ast.parse(content, filename=name)
        except (ValueError, SyntaxError) as exc:
            problems.append(str(exc))
        return {
            "path": name,
            "bytes": len(content.encode("utf-8")),
            "words": len(content.split()),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "problems": problems,
        }

    def context(self, paths: list[str], limit: int) -> str:
        """Fail rather than letting a reviewer approve unseen truncated artifacts."""
        parts = []
        for name in dict.fromkeys(paths):
            try:
                content = self.read(name)
            except (OSError, ValueError) as exc:
                content = f"[UNAVAILABLE: {exc}]"
            parts.append(f"FILE {name}\n{content}\nEND FILE {name}")
        result = "\n\n".join(parts)
        if len(result) > limit:
            raise ValueError(
                f"Artifact context needs {len(result)} characters, exceeding context_chars={limit}. "
                "Increase context_chars and num_ctx in the run configuration before resuming."
            )
        return result
