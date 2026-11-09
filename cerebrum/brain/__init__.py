"""A complete spiking-architecture brain, one module per anatomical system.

The package is organised so that every system can be removed and the rest
still runs. That is what makes the ablation experiments meaningful: the
comparison is the same brain with one part bypassed, not a different
programme.

=========================  ==========================================
module                     what it is
=========================  ==========================================
:mod:`~.sheet`             two-dimensional neuron sheet (the substrate)
:mod:`~.thalamus`          relay + reticular gate
:mod:`~.cortex`            L4 / L2-3 / L5 + prefrontal working memory
:mod:`~.basal_ganglia`     D1 / D2 / GPe / STN / GPi action selection
:mod:`~.hippocampus`       EC / DG / CA3 / CA1 with cortical reinstatement
:mod:`~.amygdala`          LA / BLA / CE, one-trial fear conditioning
:mod:`~.hypothalamus`      hunger, fatigue, fear, orexin
:mod:`~.cerebellum`        granule / Purkinje / deep nuclei, timing by LTD
:mod:`~.neuromodulation`   DA, 5-HT, NE, ACh, orexin
:mod:`~.body`              grid world with food and a predator
:mod:`~.whole_brain`       the assembly
=========================  ==========================================

Level of description differs by system on purpose, and every module says so
in its own docstring: the sheets, thalamus, amygdala, hippocampus and
cerebellum are spiking; the basal ganglia, hypothalamus, prefrontal cortex and
neuromodulators are rate-based. Selection, drive and concentration are
settling processes and slow variables, and modelling them as spikes would add
cost without adding a mechanism.
"""

from .amygdala import Amygdala
from .basal_ganglia import BasalGanglia
from .body import Body, ACTIONS, N_ACTIONS, N_SENSE
from .cerebellum import Cerebellum
from .cortex import Cortex, Prefrontal
from .hippocampus import Hippocampus
from .hypothalamus import Hypothalamus
from .neuromodulation import Neuromodulators
from .sheet import NeuronSheet, build_projection
from .thalamus import Thalamus
from .whole_brain import WholeBrain, SYSTEMS

__all__ = [
    "NeuronSheet", "build_projection",
    "Thalamus", "Cortex", "Prefrontal", "BasalGanglia", "Hippocampus",
    "Amygdala", "Hypothalamus", "Cerebellum", "Neuromodulators",
    "Body", "ACTIONS", "N_ACTIONS", "N_SENSE", "WholeBrain", "SYSTEMS",
]
