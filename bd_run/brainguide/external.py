"""Cross-project imports, kept in ONE place.

Answers the design question "can this folder access files from other projects?"
Yes — Python can import from anywhere. But instead of scattering
``sys.path.append('/Users/.../AI work/SomeRepo')`` through the code (which makes
a run non-reproducible on a VM), every cross-project reference is resolved here,
relative to the repo root or an absolute path taken from config.

Recommended layout: clone or **symlink** the external repos into the empty
placeholder folders that already exist in this tree, e.g.

    ln -s /abs/path/to/universal-brain-encoder  universal-brain-encoder-code

so imports use a stable in-repo path and the heavy weights stay .gitignored.
"""
import os
import sys
from pathlib import Path

# .../MindReading  (this file is bd_run/brainguide/external.py)
REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve(path) -> Path:
    """Absolute path, expanding ~ and resolving repo-relative paths."""
    p = Path(os.path.expanduser(str(path)))
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def add_to_path(path, label: str = "") -> Path:
    """Put an external code directory on sys.path so it can be imported."""
    p = resolve(path)
    if not p.exists():
        raise FileNotFoundError(
            f"[external] {label or 'path'} not found: {p}\n"
            f"  Vendor or symlink the code there, e.g.:\n"
            f"    ln -s /abs/path/to/repo {p}"
        )
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
    return p
