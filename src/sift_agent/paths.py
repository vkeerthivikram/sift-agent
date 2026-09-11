"""Filesystem helpers for run isolation."""

from __future__ import annotations

from itertools import count
from pathlib import Path


def unique_dir(base: Path) -> Path:
    """Atomically reserve ``base`` (or ``base-2``, ``base-3``, ... if taken).

    Uses ``mkdir(exist_ok=False)`` so concurrent processes or threads racing on
    the same timestamped name each get their own directory.
    """
    for n in count(1):
        cand = base if n == 1 else base.with_name(f"{base.name}-{n}")
        try:
            cand.mkdir(parents=True, exist_ok=False)
            return cand
        except FileExistsError:
            continue
