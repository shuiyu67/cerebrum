"""cerebrum — a spiking brain engine.

Every mechanism here changes one thing: synaptic weight. There is no loss
function, no backward pass, and no gradient anywhere in the engine.

Quick start::

    from cerebrum.core import Brain
    from cerebrum.plasticity import Eligibility, Dopamine

    brain = Brain(neurons=139_000, inputs=363, outputs=3, fan_in=360, seed=1)
    brain.add_rule(Eligibility())
    brain.add_rule(Dopamine())
    brain.run(2500)
    action, spikes = brain.step(sensors)

What v2 added over v1: inhibitory interneurons, decaying synaptic current, a
real recurrent attractor in CA3 (replacing a lookup table), a hippocampal
projection with both mossy and direct paths, homeostasis that runs at its
declared time constant, and a closed loop to play snake in.
"""

__version__ = "2.0.0"
