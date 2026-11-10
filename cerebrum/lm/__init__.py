"""A language model built on a million spiking neurons.

The pool is fixed and random; only a linear readout is trained. See
:mod:`cerebrum.lm.reservoir` for why that is the honest design for an engine
with no backward pass, and :mod:`cerebrum.lm.train` for the experiment.
"""

from .data import Corpus
from .model import ReservoirLM
from .reservoir import SparseReservoir

__all__ = ["Corpus", "ReservoirLM", "SparseReservoir"]
