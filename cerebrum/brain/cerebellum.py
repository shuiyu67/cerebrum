"""Cerebellum: the part that learns *when*.

The cortex proposes actions and the basal ganglia pick one, but neither of
them knows that a movement is late, or that the same movement will be late
again next time. Timing has to be learned from error, and that is the
cerebellum's job.

Circuit:

* **Mossy fibres** carry an efference copy of what the cortex just commanded,
  plus the sensory context.
* **Granule cells** expand that into a much larger, decorrelated code. The
  expansion is the same trick the dentate gyrus uses: it makes it possible
  for a small set of weights to pick out one moment in a sequence.
* **Purkinje cells** read the granule code through parallel fibres. They are
  the only output of the cerebellar cortex, and they are inhibitory.
* **Climbing fibres** carry the error. A climbing-fibre signal arriving
  together with parallel-fibre activity depresses those specific synapses
  (LTD). Over trials the Purkinje cell learns exactly which granule pattern
  predicts the error, and stops firing for it.
* **Deep nuclei** are held down by Purkinje cells and driven by mossy
  collaterals. Removing inhibition at the right moment is what produces a
  correctly timed conditioned response.

Values here are smaller than the rest of the brain on purpose: the cerebellar
cortex has more cells than the rest of the brain combined in mammals, but a
few thousand granule cells is enough to show the learning rule working.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .sheet import NeuronSheet


class Cerebellum:
    """Granule-Purkinje-deep nuclei with climbing-fibre LTD."""

    def __init__(self, n_mossy: int = 64, *, n_granule: int = 600,
                 n_purkinje: int = 64, n_deep: int = 64,
                 ltd: float = 0.02, seed: int = 0) -> None:
        self.n_mossy = int(n_mossy)
        self.ltd = float(ltd)
        rng = np.random.default_rng(seed + 51)

        self.granule = NeuronSheet(
            "cb_granule", n_granule, excitatory_frac=0.9, p_local=0.25,
            sigma_conn=0.1, target_hz=6.0, w_norm=0.8, th_max=4.0,
            global_inh=25.0, seed=seed + 52,
        )
        self.deep = NeuronSheet(
            "cb_deep", n_deep, excitatory_frac=0.9, p_local=0.3,
            sigma_conn=0.2, target_hz=6.0, w_norm=0.6, th_max=4.0,
            global_inh=20.0, seed=seed + 53,
        )

        # Mossy -> granule: sparse expansion.
        self.mossy_granule = (rng.random((n_granule, self.n_mossy))
                              < 0.25).astype(np.float32) * 1.1
        # Mossy -> deep: the collateral that can drive a response on its own.
        self.mossy_deep = (rng.random((n_deep, self.n_mossy))
                           < 0.35).astype(np.float32) * 1.0
        # Parallel fibres: the plastic site.
        self.pf = (rng.random((n_purkinje, n_granule))
                   < 0.05).astype(np.float32) * 0.9
        # Purkinje -> deep: inhibitory, one-to-one-ish.
        self.pc_deep = np.zeros((n_deep, n_purkinje), np.float32)
        for j in range(min(n_deep, n_purkinje)):
            self.pc_deep[j, j] = -3.0

        self.purkinje_rate = np.zeros(n_purkinje, np.float32)
        self.error_ema = 0.0
        self.trials = 0
        self.pf_delta = 0.0

    def step(self, mossy: np.ndarray, *, error: float = 0.0) -> dict:
        """One millisecond. ``error`` is the climbing-fibre signal (0..1)."""
        mossy = np.asarray(mossy, np.float32).ravel()
        if mossy.size != self.n_mossy:
            raise ValueError(f"cerebellum expects {self.n_mossy} mossy fibres")

        g_ext = (self.mossy_granule @ mossy) * 1.6
        g_idx = self.granule.step(g_ext)

        # Purkinje output is read as a rate: LTD is a slow weight change over
        # many trials, so the momentary spike pattern of a Purkinje cell
        # matters much less than its average firing.
        pf_drive = self.pf @ self.granule.rate_ema
        self.purkinje_rate = 1.0 / (1.0 + np.exp(-8.0 * (pf_drive - 0.35)))

        if error > 0.0:
            self._ltd(error)

        inh = self.pc_deep @ self.purkinje_rate
        d_ext = (self.mossy_deep @ mossy) * 1.6 + inh
        d_idx = self.deep.step(d_ext.astype(np.float32))

        self.error_ema = 0.99 * self.error_ema + 0.01 * float(error)
        return {"granule": g_idx, "deep": d_idx,
                "purkinje": self.purkinje_rate.copy(),
                "output": float(self.deep.rate_hz())}

    def _ltd(self, error: float) -> None:
        """Depress parallel fibres that were active when the error arrived.

        Only those synapses. A global depression would just make the
        cerebellum quieter; depressing the ones that *predicted the error* is
        what makes the next attempt different.
        """
        active_pf = self.granule.rate_ema > 0.01
        if not active_pf.any():
            return
        k = max(1, int(0.35 * self.pf.shape[0]))
        targets = np.argpartition(self.purkinje_rate, -k)[-k:]
        before = float(self.pf[np.ix_(targets, np.nonzero(active_pf)[0])].sum())
        scale = 1.0 - np.clip(self.ltd * float(error), 0.0, 0.2)
        self.pf[np.ix_(targets, np.nonzero(active_pf)[0])] *= scale
        after = float(self.pf[np.ix_(targets, np.nonzero(active_pf)[0])].sum())
        self.pf_delta = before - after
        self.trials += 1

    def reset(self) -> None:
        self.granule.reset()
        self.deep.reset()

    def summary(self) -> str:
        return (f"Cerebellum(granule {self.granule.n} @ {self.granule.rate_hz():.1f} Hz, "
                f"deep {self.deep.rate_hz():.1f} Hz, error {self.error_ema:.3f})")
