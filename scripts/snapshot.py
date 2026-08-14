"""Freeze the current source tree into snapshots/<name>.zip.

Stands in for `git tag` on machines where version control isn't available. Each
snapshot is an inert archive (never importable, never on sys.path) so a frozen
baseline can't accidentally shadow the live code.

    python scripts/snapshot.py v1.0-baseline-rag
    python scripts/snapshot.py exp-003-hybrid --note "RRF k=60, candidate_k=20"

Restoring is a plain unzip into an empty directory — no tooling required.
"""
import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "snapshots"

# Directories skipped wherever they appear: build output, dependencies, caches.
SKIP_DIRS = {
    ".venv", "node_modules", "dist", ".vite", "__pycache__", ".git",
    ".pytest_cache", ".ruff_cache", "snapshots", ".claude",
}

# Never archive secrets or compiled artifacts. `.env.example` is kept (it's the
# tracked template); `.env` itself holds the live API key and must not travel.
SKIP_FILES = {".env"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".zip"}

# Corpus PDFs are fetched from a manifest, not stored — keeps snapshots small.
SKIP_GLOBS = ("backend/evaluation/corpus/*.pdf",)


def _included_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if SKIP_DIRS & set(rel.parts):
            continue
        if path.name in SKIP_FILES or path.suffix in SKIP_SUFFIXES:
            continue
        if any(rel.match(pattern) for pattern in SKIP_GLOBS):
            continue
        files.append(path)
    return sorted(files)


def create_snapshot(name: str, note: str | None = None) -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    archive = OUT_DIR / f"{name}.zip"
    if archive.exists():
        # Snapshots are immutable: a phase boundary happened once. Overwriting
        # one would silently invalidate every experiment that cites it.
        sys.exit(f"Refusing to overwrite existing snapshot: {archive.name}")

    files = _included_files()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(ROOT).as_posix())

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "bytes": archive.stat().st_size,
        "sha256": digest,
        "note": note,
    }
    (OUT_DIR / f"{name}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"{archive.name}  ({len(files)} files, {archive.stat().st_size:,} bytes)")
    print(f"sha256: {digest}")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="snapshot name, e.g. v1.0-baseline-rag")
    parser.add_argument("--note", help="what this snapshot represents")
    args = parser.parse_args()
    create_snapshot(args.name, args.note)


if __name__ == "__main__":
    main()
