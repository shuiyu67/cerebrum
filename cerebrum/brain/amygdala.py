"""Amygdala: emotional tagging, and why one bad event is enough.

Fear learning breaks the rules the rest of the cortex follows. Everything
else in this brain needs repetition: the cortex shifts its weights a little
each time, the cerebellum needs hundreds of paired trials. The amygdala
learns a tone-shock association in *one* pairing and it does not decay
quickly.

The mechanism is a synaptic tag. Coincidence of a conditioned stimulus with
an unconditioned stimulus opens a window in which the CS synapse can be
rewritten in a single shot, and the window is opened by neuromodulators —
noradrenaline and acetylcholine, both released by the shock itself. That is
why the same pairing without arousal does nothing.

Circuit:

* **LA** (lateral) — where CS and US converge; holds the association.
* **BLA** (basolateral) — adds context and feeds the value back to cortex.
* **CE** (central) — the output. Drives the hypothalamus (freeze, flee,
  tachycardia) and the brainstem, and raises noradrenaline so the whole
  brain learns faster for the next few minutes.

The association itself is an explicit weight vector rather than a spiking
sheet. Synaptic tagging is a molecular switch, not a spike-timing rule, and
modelling it as an instantaneous weight write is truer to the biology than
pretending an STDP window reproduces it. The spiking sheets supply the
dynamics and the output.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .sheet import NeuronSheet


class Amygdala:
    """Associative fear with one-trial acquisition."""

    def __init__(self, n_inputs: int = 16, *, n_la: int = 120,
                 n_ce: int = 60, lr: float = 0.55,
                 extinction_lr: float = 3e-4, seed: int = 0) -> None:
        self.n_in = int(n_inputs)
        self.lr = float(lr)
        self.extinction_lr = float(extinction_lr)
        rng = np.random.default_rng(seed + 41)

        # The CS->LA association. Starts near zero: fear is learned, not
        # innate. An innate prior here would make every experiment about the
        # prior instead of about the learning.
        self.w_cs = np.zeros(self.n_in, np.float32)
        self.w_context = np.zeros(self.n_in, np.float32)

        self.la = NeuronSheet(
            "amy_LA", n_la, excitatory_frac=0.85, p_local=0.3,
            sigma_conn=0.15, target_hz=4.0, w_norm=0.6, th_max=4.0,
            global_inh=15.0, seed=seed + 42,
        )
        self.ce = NeuronSheet(
            "amy_CE", n_ce, excitatory_frac=0.9, p_local=0.3,
            sigma_conn=0.2, target_hz=6.0, w_norm=0.5, th_max=4.0,
            global_inh=15.0, seed=seed + 43,
        )
        self.la_to_ce = (rng.random((n_ce, n_la)) < 0.35).astype(np.float32) * 1.2

        self.fear = 0.0
        self.acquired = 0.0
        self.trials = 0
        self.pairings = 0

    def step(self, cs: np.ndarray, us: float = 0.0, *,
             arousal: float = 1.0) -> dict:
        """One millisecond. ``us`` is the aversive event (0..1)."""
        cs = np.asarray(cs, np.float32).ravel()
        pred = float(self.w_cs @ cs)

        la_ext = (pred * 1.1 + us * 2.2) * arousal
        la_idx = self.la.step(np.full(self.la.n, la_ext, np.float32))

        syn = (self.la_to_ce[:, la_idx].sum(axis=1)
               if la_idx.size else None)
        ce_idx = self.ce.step(None, syn_extra=syn)

        # Extinction: the cue is present and nothing bad happens, so the
        # association weakens. Without it a single pairing generalises to
        # every situation that shares the cue, fear never comes down, and the
        # animal does nothing but flee — measured, 3000 of 3000 steps spent
        # on the flee action. Acquisition is one trial; extinction is slow,
        # which is the asymmetry that makes fear hard to unlearn.
        if us <= 0.0 and self.w_cs.sum() > 0:
            self.w_cs *= (1.0 - self.extinction_lr)
        self.fear = 0.9 * self.fear + 0.1 * min(1.2, la_ext * 0.45 + us)
        return {"la": la_idx, "ce": ce_idx, "fear": float(self.fear),
                "prediction": pred}

    def learn(self, cs: np.ndarray, us: float, *, arousal: float = 1.0,
              ach: float = 1.0) -> float:
        """Tag the CS synapse if the US is present.

        ``arousal`` and ``ach`` are the gate. With arousal at zero the
        pairing does nothing at all — which is the correct behaviour, and the
        control condition for the one-shot claim.
        """
        self.trials += 1
        if us <= 0.0:
            return 0.0
        cs = np.asarray(cs, np.float32).ravel()
        gain = self.lr * float(us) * float(arousal) * float(ach)
        self.pairings += 1
        delta = gain * (cs - self.w_cs)
        self.w_cs = np.clip(self.w_cs + delta, 0.0, 3.0).astype(np.float32)
        self.acquired = float(np.linalg.norm(self.w_cs))
        return float(delta.max()) if delta.size else 0.0

    def reset(self) -> None:
        self.la.reset()
        self.ce.reset()

    def summary(self) -> str:
        return (f"Amygdala(LA {self.la.n} @ {self.la.rate_hz():.1f} Hz, "
                f"fear {self.fear:.2f}, {self.pairings} pairings)")
