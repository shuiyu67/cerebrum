"""Closed-loop control tasks for the spiking pool."""
from .pendulum import InvertedPendulum, PoolController, pd_teacher

__all__ = ["InvertedPendulum", "PoolController", "pd_teacher"]
