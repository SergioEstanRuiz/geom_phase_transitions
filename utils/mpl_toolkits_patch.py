"""Utilities to make ``mpl_toolkits`` pick the pip version of Matplotlib."""

from __future__ import annotations

from pathlib import Path
from typing import List


def ensure_pip_namespace_first() -> bool:
    """Put the pip ``mpl_toolkits`` directory ahead of the system one."""
    try:
        import matplotlib  # type: ignore
        import mpl_toolkits  # type: ignore
    except Exception:
        return False

    pip_dir = (
        Path(matplotlib.__file__).resolve().parent.parent / "mpl_toolkits"
    )
    if not pip_dir.is_dir():
        return False

    pip_path = str(pip_dir)
    current: List[str] = list(getattr(mpl_toolkits, "__path__", []))

    if current[:1] == [pip_path]:
        return True

    if pip_path in current:
        current.remove(pip_path)
    current.insert(0, pip_path)
    mpl_toolkits.__path__ = current
    return True
