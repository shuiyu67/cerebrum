"""Hypothalamus: the drives that make behaviour have a reason.

Everything built so far is reactive. A stimulus arrives, a loop settles, an
action comes out. Nothing in that arrangement *wants* anything, so the system
has no basis for choosing between two equally available actions and no reason
to stop resting.

The hypothalamus supplies the missing quantity: internal state. It is the
part of the brain that knows the body is short of energy, short of sleep, or
in danger, and it turns those facts into signals the rest of the system can
act on.

Drives implemented:

* **hunger** — rises as energy falls, relieved by eating.
* **fatigue** — rises with exertion and time awake, relieved by rest.
* **fear** — driven by the amygdala, decays over seconds.
* **temperature** — a slow variable for completeness; it modulates arousal
  rather than gating an action of its own.

Outputs:

* **orexin** — arousal. High orexin raises cortical gain and keeps the animal
  awake; a fatigued animal with low orexin loses cortical responsiveness,
  which is what sleep pressure feels like from the inside.
* **forage bias** — pushes action selection towards exploration and eating.
* **rest bias** — pushes towards stopping.
* **CRF** — stress; raises noradrenaline and biases the basal ganglia
  towards the indirect (no-go) pathway, i.e. freezing.

This module is a set of leaky scalars, not a spiking sheet, and that is the
right level: hypothalamic drive states are slow humoral variables lasting
minutes to hours. Their firing-rate correlates matter far less than their
time constants.
"""

from __future__ import annotations

import numpy as np


class Hypothalamus:
    """Homeostatic drive variables and their descending outputs."""

    def __init__(self, *, hunger_rate: float = 0.0009,
                 fatigue_rate: float = 0.0006, seed: int = 0) -> None:
        self.hunger = 0.35
        self.fatigue = 0.10
        self.fear = 0.0
        self.temperature = 0.5
        self.orexin = 0.7

        self.hunger_rate = float(hunger_rate)
        self.fatigue_rate = float(fatigue_rate)
        self.energy = 1.0
        self.steps = 0

    # -- body feedback ---------------------------------------------------

    def eat(self, amount: float = 0.55) -> None:
        self.hunger = float(np.clip(self.hunger - amount, 0.0, 2.0))
        self.energy = float(np.clip(self.energy + amount, 0.0, 1.5))

    def exert(self, amount: float = 0.02) -> None:
        self.energy = float(np.clip(self.energy - amount, 0.0, 1.5))
        self.fatigue = float(np.clip(self.fatigue + amount * 0.6, 0.0, 2.0))

    def rest(self, amount: float = 0.01) -> None:
        self.fatigue = float(np.clip(self.fatigue - amount, 0.0, 2.0))
        self.energy = float(np.clip(self.energy + amount * 0.5, 0.0, 1.5))

    # -- dynamics --------------------------------------------------------

    def observe(self, sensors: np.ndarray) -> None:
        """Read interoception: hunger, fatigue and energy come from the body.

        Without this the drives are open-loop. Measured: hypothalamic
        fatigue rose monotonically at its own rate because nothing ever told
        it the animal had rested, so after a few thousand steps the rest
        bias dominated every other consideration and the animal sat still
        until it starved. A drive that cannot be relieved is not a drive, it
        is a timer.
        """
        s = np.asarray(sensors, np.float32).ravel()
        if s.size >= 11:
            self.hunger = float(np.clip(0.5 * self.hunger + 0.5 * s[8], 0.0, 1.6))
            self.fatigue = float(np.clip(0.5 * self.fatigue + 0.5 * s[9], 0.0, 1.6))
            self.energy = float(np.clip(s[10], 0.0, 1.5))

    def step(self, *, fear_in: float = 0.0, dt: float = 1.0) -> dict:
        """Advance 1 ms of hypothalamic state."""
        self.steps += 1

        # hunger tracks the energy deficit rather than climbing blindly, so
        # eating actually shuts it off instead of merely slowing it.
        deficit = max(0.0, 1.0 - self.energy)
        self.hunger += (deficit * self.hunger_rate
                        - 0.0004 * (self.hunger - deficit)) * dt
        self.hunger = float(np.clip(self.hunger, 0.0, 1.6))

        self.fatigue = float(np.clip(
            self.fatigue + (self.fatigue_rate * (1.0 - self.fatigue * 0.4)) * dt,
            0.0, 1.6))

        # fear is driven by the amygdala and decays with a few-second time
        # constant; a fear that never decayed would mean an animal that never
        # leaves cover.
        self.fear += (float(fear_in) - self.fear) * 0.004 * dt

        # orexin: wakefulness. Suppressed by fatigue, raised by hunger and
        # fear. This is the link that lets a hungry animal override tiredness.
        target = 0.75 - 0.9 * self.fatigue + 0.35 * self.hunger + 0.5 * self.fear
        self.orexin += (float(np.clip(target, 0.0, 1.4)) - self.orexin) * 0.002 * dt

        crf = float(np.clip(0.7 * self.fear + 0.2 * self.hunger, 0.0, 1.0))

        return {
            "hunger": self.hunger,
            "fatigue": self.fatigue,
            "fear": self.fear,
            "orexin": self.orexin,
            "crf": crf,
            # Fear and hunger drive *different* actions, so fear damps
            # foraging rather than cancelling it. With a 0.5 coefficient a
            # scared animal scored 0 on forage and never ate again.
            "forage_bias": float(np.clip(self.hunger - 0.25 * self.fear,
                                         -1.0, 1.0)),
            "rest_bias": float(np.clip(self.fatigue - 0.3 * self.hunger, -1.0, 1.0)),
            "arousal": float(np.clip(self.orexin * (1.0 - 0.6 * self.fatigue), 0.1, 1.5)),
        }

    def summary(self) -> str:
        return (f"Hypothalamus(hunger {self.hunger:.2f}, fatigue {self.fatigue:.2f}, "
                f"fear {self.fear:.2f}, orexin {self.orexin:.2f})")
