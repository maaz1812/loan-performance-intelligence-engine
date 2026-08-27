"""
Raw ZIP access -- streaming, never extracting.

dataset_usage.md Section 17.3 Rule 1 forbids opening the raw files manually or
loading a complete CSV into memory. Rule 2 requires chunk-based processing.
This module is the only place in the codebase that touches data/raw/.

Design constraint that drove this module: the 16 archives total ~15 GB
compressed but ~200 GB uncompressed against 56 GB of free disk, so extraction
to disk is not an option. Every member is streamed through
`zipfile.ZipFile.open()`, which yields a file-like object that pandas' C parser
can consume incrementally. Peak memory is one chunk, not one file.

A useful side effect: because nothing is ever written back, the raw
immutability guarantee is structural rather than procedural. `verify_immutable`
records SHA-256 checksums so it can be proven after the fact.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

from ..config import RAW_DIR, REFERENCE_DIR, raw_zips


@dataclass
class ArchiveInfo:
    """Metadata for one raw quarterly archive."""
    zip_name: str
    member_name: str
    quarter: str
    compressed_bytes: int
    uncompressed_bytes: int
    compression_ratio: float

    @property
    def uncompressed_gb(self) -> float:
        return self.uncompressed_bytes / 1e9


def describe_archive(zip_path: Path) -> ArchiveInfo:
    """Read archive metadata from the ZIP central directory -- no decompression."""
    with zipfile.ZipFile(zip_path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if len(infos) != 1:
            names = [i.filename for i in infos]
            raise ValueError(
                f"{zip_path.name}: expected exactly one CSV member, found {names}"
            )
        info = infos[0]
    return ArchiveInfo(
        zip_name=zip_path.name,
        member_name=info.filename,
        quarter=zip_path.stem,
        compressed_bytes=info.compress_size,
        uncompressed_bytes=info.file_size,
        compression_ratio=(info.file_size / info.compress_size) if info.compress_size else 0.0,
    )


def describe_all() -> list[ArchiveInfo]:
    return [describe_archive(p) for p in raw_zips()]


def open_member(zip_path: Path):
    """
    Open the single CSV member of `zip_path` as a binary stream.

    Returns (zipfile_handle, member_stream). Both must be closed by the caller;
    `stream_member` below is the context-managed form and should be preferred.
    """
    zf = zipfile.ZipFile(zip_path)
    info = describe_archive(zip_path)
    return zf, zf.open(info.member_name, "r")


class stream_member:
    """
    Context manager yielding a binary stream over a ZIP's CSV member.

        with stream_member(path) as fh:
            for chunk in pd.read_csv(fh, sep="|", chunksize=500_000, ...):
                ...

    Nothing is written to disk at any point.
    """

    def __init__(self, zip_path: Path):
        self.zip_path = Path(zip_path)
        self._zf: zipfile.ZipFile | None = None
        self._fh = None

    def __enter__(self):
        self._zf = zipfile.ZipFile(self.zip_path)
        info = describe_archive(self.zip_path)
        self._fh = self._zf.open(info.member_name, "r")
        return self._fh

    def __exit__(self, *exc):
        if self._fh is not None:
            self._fh.close()
        if self._zf is not None:
            self._zf.close()
        return False


def peek_lines(zip_path: Path, n: int = 5) -> list[str]:
    """Read the first n lines of a member for inspection (bounded read)."""
    out: list[str] = []
    with stream_member(zip_path) as fh:
        buf = fh.read(64 * 1024).decode("utf-8", errors="replace")
    for line in buf.split("\n")[:n]:
        out.append(line)
    return out


def sha256_of(path: Path, block: int = 8 * 1024 * 1024) -> str:
    """Streaming SHA-256 so a 2 GB archive never lands in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_immutable(write_manifest: bool = True) -> dict:
    """
    Checksum every raw archive and compare against a stored manifest.

    First run establishes the baseline. Later runs prove the raw layer was not
    modified -- the auditable-lineage requirement of dataset_usage.md
    Sections 3.1 and 16.
    """
    manifest_path = REFERENCE_DIR / "raw_manifest.json"
    current = {}
    for p in raw_zips():
        info = describe_archive(p)
        current[p.name] = {
            "sha256": sha256_of(p),
            "compressed_bytes": info.compressed_bytes,
            "uncompressed_bytes": info.uncompressed_bytes,
            "member": info.member_name,
        }

    result: dict = {"status": "baseline_created", "files": len(current), "changed": []}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        changed = [
            name for name, meta in current.items()
            if name in previous and previous[name]["sha256"] != meta["sha256"]
        ]
        missing = [n for n in previous if n not in current]
        result = {
            "status": "verified" if not changed and not missing else "MODIFIED",
            "files": len(current),
            "changed": changed,
            "missing": missing,
        }
    elif write_manifest:
        manifest_path.write_text(json.dumps(current, indent=2))

    if write_manifest and not manifest_path.exists():
        manifest_path.write_text(json.dumps(current, indent=2))
    return result


def corpus_summary() -> dict:
    """Aggregate size figures used in the reports."""
    infos = describe_all()
    total_c = sum(i.compressed_bytes for i in infos)
    total_u = sum(i.uncompressed_bytes for i in infos)
    return {
        "n_archives": len(infos),
        "compressed_gb": round(total_c / 1e9, 2),
        "uncompressed_gb": round(total_u / 1e9, 2),
        "mean_compression_ratio": round(total_u / total_c, 2) if total_c else 0.0,
        "largest_member_gb": round(max((i.uncompressed_gb for i in infos), default=0), 2),
        "archives": [asdict(i) for i in infos],
    }


__all__ = [
    "ArchiveInfo", "describe_archive", "describe_all", "open_member",
    "stream_member", "peek_lines", "sha256_of", "verify_immutable",
    "corpus_summary",
]
