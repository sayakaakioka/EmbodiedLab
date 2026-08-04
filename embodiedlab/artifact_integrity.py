"""Integrity metadata for downloadable artifact files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

HASH_CHUNK_SIZE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class FileIntegrity:
    """Exact byte size and lowercase SHA-256 digest of one file."""

    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, int | str]:
        """Return JSON-compatible integrity metadata."""
        return asdict(self)


def compute_file_integrity(path: str | Path) -> FileIntegrity:
    """Stream one file once to calculate its size and SHA-256 digest."""
    file_path = Path(path)
    digest = sha256()
    size_bytes = 0
    with file_path.open("rb") as source:
        while chunk := source.read(HASH_CHUNK_SIZE_BYTES):
            size_bytes += len(chunk)
            digest.update(chunk)
    return FileIntegrity(
        size_bytes=size_bytes,
        sha256=digest.hexdigest(),
    )
