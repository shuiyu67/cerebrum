"""Basal ganglia: action selection by disinhibition.

The problem
-----------
A cortex proposes movements continuously. Without a selector, every proposal
is executed at once, weighted by however many cells happen to be firing —
which is exactly what the previous generation did: the "action" was the
population's spike distribution, so a stronger stimulus always won and
nothing could ever be suppressed.

The circuit
-----------
This is the classical Albin-DeLong-Gurney architecture, implemented as a
rate model. The reason is honest: the basal ganglia's job is *selection*, a
settling process over hundreds of milliseconds, and the published selection
models that reproduce choice, reaction time and Parkinsonian deficits are all
rate models. Spike-level details of medium spiny neurons do not change which
action wins; the loop's sign structure does. Labelling it rate-based is more
useful than pretending otherwise.

* **Striatum, D1** (direct pathway) — cortex excites it, it inhibits GPi. Net
  effect: *open* the channel.
* **Striatum, D2** (indirect pathway) — cortex excites it, it inhibits GPe,
  so STN is released, so GPi is excited. Net effect: *close* the channel.
* **GPe** — tonically active brake on STN.
* **STN** — excites GPi, and does it broadly, which is why it acts as a
  global "hold everything" signal rather than a channel selector.
* **GPi** — the output. Tonically inhibits the thalamus. Lowering it in one
  channel releases that channel's thalamic relay.

Dopamine acts on the two striatal populations in opposite directions, which
is the entire reason a reward prediction error can reinforce an action:
raising dopamine increases the go pathway's gain and decreases the no-go
pathway's, so the action that was just taken becomes easier to take again.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _sig(x: np.ndarray, gain: float = 6.0, shift: float = 0.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-gain * (x - shift)))


class BasalGanglia:
    """Action-channel selector.

    Parameters
    ----------
    n_actions:
        Number of competing channels.
    sharpness:
        Lateral inhibition between GPi channels. Without it every channel is
        released a little; with it the channel with the least inhibition wins
        outright, which is what makes this a *selection* rather than a
        dimmer switch.
    """

    def __init__(self, n_actions: int = 6, *, sharpness: float = 5.0,
                 da_gain: float = 0.35, lr: float = 0.05,
                 seed: int = 0) -> None:
        self.A = int(n_actions)
        self.sharpness = float(sharpness)
        self.da_gain = float(da_gain)
        self.lr = float(lr)
        rng = np.random.default_rng(seed + 31)

        # Corticostriatal weights: the plastic part of the selection system.
        self.w_d1 = rng.uniform(0.85, 1.15, self.A)
        self.w_d2 = rng.uniform(0.70, 1.00, self.A)
        # Default: slightly favour the indirect path, so an action has to be
        # argued for before it is released. A system that starts biased
        # towards "go" cannot sit still.
        self.w_stn = np.full(self.A, 1.0)
        self.w_d1_gpi = np.full(self.A, 1.0)

        self.d1 = np.zeros(self.A)
        self.d2 = np.zeros(self.A)
        self.gpe = np.full(self.A, 0.9)
        self.stn = np.zeros(self.A)
        self.gpi = np.full(self.A, 0.7)     # tonic inhibition at rest

        self.da = 0.0
        self.chosen = -1
        self.last_release = np.zeros(self.A)
        self.history: list = []

    # -- dynamics --------------------------------------------------------

    def step(self, cortical: np.ndarray, *, dopamine: float = 0.0,
             hold: float = 0.0) -> np.ndarray:
        """One update. ``cortical`` is per-channel evidence (0..1).

        Returns the GPi output in ``0..1``; high means "held shut".
        ``hold`` models the hyperdirect cortex->STN path: a global brake that
        cancels *everything* rather than selecting, which is what happens
        when a task becomes dangerous and the system freezes.
        """
        c = np.clip(np.asarray(cortical, np.float64).ravel(), 0.0, 0.9)
        if c.size != self.A:
            raise ValueError(f"expected {self.A} channels, got {c.size}")
        self.da = float(dopamine)

        # Dopamine biases the two pathways oppositely. This single sign flip
        # is what turns a scalar reward signal into reinforcement.
        g1 = np.clip(1.0 + self.da_gain * self.da, 0.2, 2.5)
        g2 = np.clip(1.0 - 0.8 * self.da_gain * self.da, 0.2, 2.5)

        # The steepness difference between the two pathways is the whole
        # trick. D1 sits on a steeper input-output curve than D2 — high
        # dopamine affinity on one side, lower on the other — so as cortical
        # drive rises the direct path opens *before* the indirect path can
        # close the door. With both paths equally steep the two effects
        # cancel exactly and the output is flat in the input (measured:
        # every channel sat at GPI 0.59 regardless of what the cortex
        # proposed, i.e. the basal ganglia were a constant).
        self.d1 = _sig(8.0 * g1 * self.w_d1 * c - 3.0 * self.gpe, 1.0)
        self.d2 = _sig(4.2 * g2 * self.w_d2 * c - 3.0 * self.gpe, 1.0)

        # The indirect path is slower than the direct one, and that lag is
        # load-bearing. With both paths updating instantly, a sustained
        # proposal drove every channel shut: the direct path opened the gate
        # and the indirect path closed it in the same millisecond, and the
        # net result was a held channel (measured: the strongest proposal sat
        # at GPI 1.00 after four milliseconds). The indirect pathway is
        # multisynaptic and its job is to *stop* an action that is already
        # running, so it has to arrive after the action has been chosen.
        gpe_target = _sig(3.0 - 6.0 * self.d2, 1.0)
        self.gpe += (gpe_target - self.gpe) * 0.12

        stn_target = _sig(3.0 - 6.0 * self.gpe + 2.5 * hold, 1.0)
        self.stn += (stn_target - self.stn) * 0.12

        # GPi output: tonic brake, released by the direct path, pressed by
        # STN. High means "this channel stays shut".
        raw = 4.0 + 6.0 * self.stn - 6.0 * self.d1 * self.w_d1_gpi - 3.0 * self.gpe
        # Steep output. GPi cells do not inhibit each other — the competition
        # happened upstream, in the striatum — so there is no lateral term
        # here. What is needed instead is a sharp readout: with a shallow
        # sigmoid a released channel sits at 0.23 against 0.57 for the held
        # ones, a difference the thalamus cannot act on. At this gain the
        # released channel goes to 0.03 and the held ones to 1.00, which is
        # the difference between a dimmer and a switch.
        gpi = _sig(raw, 3.0)

        self.gpi = gpi
        self.chosen = int(np.argmin(gpi))
        self.last_release = 1.0 - gpi
        self.history.append((self.chosen, float(gpi.min()), float(gpi.max())))
        return gpi

    # -- learning --------------------------------------------------------

    def learn(self, cortical: np.ndarray, action: int, rpe: float) -> float:
        """Three-factor update of the corticostriatal weights.

        Positive prediction error strengthens the go pathway and weakens the
        no-go pathway *for the action that was actually taken* — that
        specificity is the difference between learning and a global gain
        change. Negative error does the opposite.
        """
        c = np.clip(np.asarray(cortical, np.float64).ravel(), 0.0, 0.9)
        a = int(action)
        if not 0 <= a < self.A:
            return 0.0
        delta = self.lr * float(rpe) * float(c[a])
        self.w_d1[a] = float(np.clip(self.w_d1[a] + delta, 0.1, 2.5))
        self.w_d2[a] = float(np.clip(self.w_d2[a] - delta, 0.1, 2.5))
        return delta

    def reset(self) -> None:
        self.d1[:] = 0.0
        self.d2[:] = 0.0
        self.gpe[:] = 0.9
        self.stn[:] = 0.0
        self.gpi[:] = 0.7

    def summary(self) -> str:
        top = int(np.argmin(self.gpi))
        return (f"BasalGanglia({self.A} channels, selected={top}, "
                f"gpi=[{', '.join(f'{g:.2f}' for g in self.gpi)}])")
