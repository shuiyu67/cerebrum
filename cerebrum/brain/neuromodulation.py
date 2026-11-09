"""Neuromodulation: five slow signals that change how the rest computes.

A brain with one reward signal can learn what to do. It cannot learn *how long
to wait*, *how fast to adapt*, *how alert to be*, or *when to stop exploring*.
Those are separate problems and they are solved by separate chemicals, each
with its own source nucleus, its own time constant, and its own target.

* **Dopamine** (VTA/SNc) — reward prediction error. Phasic, fast. Gates
  corticostriatal plasticity and opens the prefrontal write gate.
* **Serotonin** (raphe) — the slow average of recent reward. Sets patience:
  high serotonin means keep doing what you are doing, low serotonin means the
  current strategy is not paying and something else should be tried.
* **Noradrenaline** (locus coeruleus) — unexpectedness. Fires for surprise
  and threat, raises cortical gain and the amygdala's learning rate, and
  shortens the time the system will tolerate no reward.
* **Acetylcholine** (basal forebrain) — expected uncertainty. Raises the
  learning rate where prediction is failing, which is attention.
* **Orexin** (hypothalamus) — wakefulness. Scales arousal, and is the reason
  a hungry animal can override being tired.

Every one of these is a scalar with first-order dynamics. That is not a
simplification of convenience: the biological signals are volume-transmitted
concentrations with time constants from hundreds of milliseconds to minutes,
and a scalar is the correct model of a concentration.
"""

from __future__ import annotations

import numpy as np


class Neuromodulators:
    """Five concentrations, their sources, and their effects."""

    def __init__(self, *, da_tau: float = 180.0, ht_tau: float = 6000.0,
                 ne_tau: float = 400.0, ach_tau: float = 900.0,
                 seed: int = 0) -> None:
        self.da = 0.0
        self.da_tonic = 0.0
        self.serotonin = 0.5
        self.ne = 0.3
        self.ach = 0.4
        self.orexin = 0.7

        self.da_tau = float(da_tau)
        self.ht_tau = float(ht_tau)
        self.ne_tau = float(ne_tau)
        self.ach_tau = float(ach_tau)

        # reward predictor (a critic, not a teacher)
        self.value = 0.0
        self.value_lr = 0.01
        self.rpe = 0.0
        self.reward_ema = 0.0
        self.recent_rpe = []

    # -- inputs ----------------------------------------------------------

    def reward(self, r: float) -> float:
        """Report a reward; returns the prediction error it produced."""
        r = float(r)
        self.rpe = r - self.value
        self.value += self.value_lr * self.rpe
        self.da += 1.6 * self.rpe
        self.da = float(np.clip(self.da, -1.5, 2.5))
        self.reward_ema = 0.995 * self.reward_ema + 0.005 * r
        self.recent_rpe.append(self.rpe)
        if len(self.recent_rpe) > 200:
            self.recent_rpe.pop(0)
        return self.rpe

    def surprise(self, amount: float) -> None:
        """Novelty or threat, from CA1 or the amygdala."""
        self.ne = float(np.clip(self.ne + 0.35 * float(amount), 0.0, 1.5))

    def fatigue_signal(self, fatigue: float, orexin_drive: float) -> None:
        self.orexin = float(np.clip(orexin_drive, 0.0, 1.4))

    # -- dynamics --------------------------------------------------------

    def step(self, *, uncertainty: float = 0.0) -> dict:
        """Advance 1 ms of concentration dynamics."""
        # phasic dopamine decays fast; its tonic level is a slow average of
        # the same signal, which is what makes it usable as a motivation
        # signal rather than a momentary pulse
        self.da *= np.exp(-1.0 / self.da_tau)
        self.da_tonic += (self.da - self.da_tonic) * 0.0008

        # serotonin tracks the slow average reward. Patience is just the
        # ratio of what you have been getting to what you expect.
        target_ht = float(np.clip(0.5 + 4.0 * self.reward_ema, 0.05, 1.2))
        self.serotonin += (target_ht - self.serotonin) * (1.0 / self.ht_tau)

        self.ne *= np.exp(-1.0 / self.ne_tau)
        self.ne = float(np.clip(self.ne + 0.001 * 0.3, 0.05, 1.5))

        # acetylcholine rises with uncertainty: prediction failing is the
        # definition of a situation worth attending to
        ach_target = 0.3 + 0.7 * float(np.clip(uncertainty, 0.0, 1.0))
        self.ach += (ach_target - self.ach) * (1.0 / self.ach_tau)

        return self.levels()

    def levels(self) -> dict:
        return {
            "da": self.da, "da_tonic": self.da_tonic,
            "serotonin": self.serotonin, "ne": self.ne,
            "ach": self.ach, "orexin": self.orexin, "rpe": self.rpe,
        }

    # -- effects ---------------------------------------------------------

    def cortical_gain(self) -> float:
        """Arousal multiplier for cortical responsiveness."""
        return float(np.clip(0.55 + 0.5 * self.ne + 0.45 * self.orexin
                             + 0.2 * self.ach, 0.3, 1.8))

    def learning_rate(self) -> float:
        """How fast the system should be rewriting itself right now."""
        return float(np.clip(0.4 + 0.8 * self.ach + 0.4 * abs(self.da)
                             + 0.3 * self.ne, 0.2, 2.5))

    def patience(self) -> float:
        """How long to keep trying before giving up on the current option."""
        return float(np.clip(0.3 + 1.4 * self.serotonin - 0.4 * self.ne,
                             0.1, 2.0))

    def summary(self) -> str:
        return (f"Modulators(DA {self.da:+.2f} 5-HT {self.serotonin:.2f} "
                f"NE {self.ne:.2f} ACh {self.ach:.2f} ORX {self.orexin:.2f})")
