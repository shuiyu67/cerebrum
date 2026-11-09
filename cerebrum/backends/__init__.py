"""Storage and compute backends.

``numpy`` is used when it is importable; ``pure`` needs nothing beyond the
standard library. Both expose the same surface:

* ``SparseSynapses`` — event-driven sparse propagation
* ``Population`` — leaky integrate-and-fire neurons
* ``Rng`` — deterministic random numbers
* ``build_random`` — vectorised/looped random projection
* ``zeros`` / ``group_counts`` — small array helpers
"""

from __future__ import annotations

from typing import Any, Optional

__all__ = ["load", "available", "pure"]


def available() -> list[str]:
    """Backend names importable in this environment."""
    names = ["pure"]
    try:
        import numpy  # noqa: F401

        names.insert(0, "numpy")
    except Exception:
        pass
    return names


def load(name: Optional[str] = None) -> Any:
    """Import and return a backend module.

    ``None`` selects NumPy when available and falls back to the standard
    library implementation.
    """
    if name in (None, "auto"):
        name = available()[0]
    if name == "numpy":
        from . import numpy_backend

        return numpy_backend
    if name == "pure":
        from . import pure

        return pure
    raise ValueError(f"unknown backend {name!r}; choose from {available()}")
