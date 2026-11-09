"""CA3 — a spiking recurrent attractor, not a lookup table.

The real CA3 is the part of the brain that performs pattern completion: a
partial cue entering from the dentate gyrus is enough to re-activate the whole
stored episode, because the CA3 cells that took part in that episode are wired
to each other through their recurrent collaterals. Two numbers matter:

* the rat has ~3x10^5 CA3 cells with ~1.2x10^4 recurrent synapses each, a
  contact probability of about **2%** (Rolls; Amaral et al.);
* a stored episode activates a sparse subset, so one *episode* cell receives
  roughly ``0.02 x 3000 ≈ 60`` inputs from cells of the same episode.

That 60 is the functional quantity: it is what lets 20% of the cue drag the
other 80% back to threshold. It is not the 2% that matters, and shrinking the
network while keeping 2% destroys the ability outright — at 2000 cells, 2%
gives each episode cell ``0.02 x 100 = 2`` same-episode inputs, far too few to
complete anything. So this implementation keeps the *functional* quantity
instead of the ratio: 2000 cells, 20% density, 100 cells per episode, and
therefore ~20 same-episode inputs per cell.

Architecture
------------
``inputs -> [mossy fibres, sparse] -> excitatory CA3 (recurrent) -> CA1``
                                      ^            |
                                      |     inhibitory pool (20%)
                                      +------------+

The inhibitory pool is not decoration. Without it the recurrent weights either
do nothing (too weak) or the whole network ignites on one cue (too strong).
Excitation and inhibition have to be balanced dynamically — this is the same
80/20 split that is invariant across cortex, and it is what the global gain
loop in :class:`~cerebrum.core.Brain` was standing in for.
"""

from __future__ import annotations

import numpy as np

from . import backends as _backends

__all__ = ["CA3Network"]


class CA3Network:
    """A spiking attractor with excitatory recurrence and feedback inhibition.

    Parameters
    ----------
    n_cells:
        Excitatory pyramidal cells.
    fan_in:
        Recurrent synapses per cell. ``fan_in / n_cells`` is the contact
        probability (0.2 here, against 0.02 in the rat — see the module
        docstring for why the ratio is not preserved).
    k:
        Cells activated by one episode. Sparse coding is what keeps episodes
        from interfering, and it sets how many same-episode inputs a cell gets
        (``fan_in / n_cells * k``).
    n_inhib:
        Inhibitory cells, 20% of the excitatory count as in cortex.
    ltp:
        Weight written per episode, in units of the threshold. A cell needs
        ``1 / ltp`` same-episode inputs to reach threshold on its own.
    """

    def __init__(
        self,
        n_cells: int = 2000,
        fan_in: int = 400,
        k: int = 100,
        n_inhib: int = 0,
        inh_fan_in: int = 400,
        seed: int = 0,
        backend: str = "numpy",
        ltp: float = 0.09,
        w_max: float = 0.12,
        cue_current: float = 2.2,
        inh_gain: float = 1.0,
        noise: float = 0.08,
        syn_decay: float = 0.85,
    ) -> None:
        self.n_cells = int(n_cells)
        self.fan_in = int(min(fan_in, n_cells - 1))
        self.k = int(min(k, n_cells))
        self.n_inhib = int(n_inhib if n_inhib else max(4, n_cells // 5))
        self.ltp = float(ltp)
        self.w_max = float(w_max)
        self.cue_current = float(cue_current)
        self.inh_gain = float(inh_gain)
        self.noise = float(noise)
        # Synaptic current has a time constant of its own. Delivering each
        # step's input as an instantaneous kick makes a spike almost invisible
        # to its target: the membrane only moves 1/tau_m of the way, then the
        # drive disappears. Measured, a 50-cell cue activated 37 cells and then
        # activity died, because no target ever accumulated enough. A decaying
        # current (AMPA-like, ~7 ms here) is what lets one volley hand over to
        # the next, and therefore what makes an attractor possible at all.
        self.syn_decay = float(syn_decay)

        b = _backends.load(backend)
        self.backend = b
        self.rng = b.Rng(seed)

        # -- excitatory pool: silent until driven, no pacemakers -----------
        self.exc = b.Population(
            self.n_cells, self.rng,
            tau_m=12.0, v_threshold=1.0, refractory_steps=2,
            g_adapt=0.15, tau_adapt=200.0,
            bias_mean=0.0, bias_std=0.01,
            pacemaker_frac=0.0, pacemaker_boost=0.0,
            u_short_term=0.05, tau_recovery=200.0,
            target_hz=3.0, eta_homeostasis=0.0,
        )
        self.exc.homeostasis_on = False

        # -- inhibitory pool ------------------------------------------------
        self.inh = b.Population(
            self.n_inhib, self.rng,
            tau_m=6.0, v_threshold=0.5, refractory_steps=1,
            g_adapt=0.0, tau_adapt=200.0,
            bias_mean=0.0, bias_std=0.01,
            pacemaker_frac=0.0, pacemaker_boost=0.0,
            u_short_term=0.0, tau_recovery=200.0,
            target_hz=3.0, eta_homeostasis=0.0,
        )
        self.inh.homeostasis_on = False

        # -- recurrent collaterals -----------------------------------------
        self.rec = b.SparseSynapses(self.n_cells, self.n_cells)
        pre, post, w = b.build_random(
            self.n_cells, self.n_cells, self.fan_in, self.rng, 1.0
        )
        self.rec.add_batch(pre, post, np.zeros(len(w), np.float32))

        # -- exc -> inh : each inhibitory cell samples the excitatory pool --
        self.e_to_i = b.SparseSynapses(self.n_cells, self.n_inhib)
        # Calibration: a CA3 cell has fan_in/n_cells * n_cells = fan_in outgoing
        # synapses spread over the inhibitory pool, so with E active cells each
        # inhibitory cell gets E * fan_in / n_cells events. For E=50, fan_in=400,
        # n_cells=2000 that is 10 events, so unit weight per synapse means the
        # whole pool ignites at 10x threshold. Scaling by 1/40 puts ignition at
        # the activity level where inhibition should start to bite.
        ei_scale = 1.0 / max(1.0, self.fan_in * 4.0 / self.n_cells * 10.0)
        pre, post, w = b.build_random(
            self.n_cells, self.n_inhib, min(inh_fan_in, self.n_cells),
            self.rng, ei_scale,
        )
        # sign matters: build_random draws from a zero-mean normal, so half
        # of these would be inhibitory if used raw. An excitatory projection
        # must be strictly positive.
        self.e_to_i.add_batch(pre, post, np.abs(w).astype(np.float32))

        # -- inh -> exc : divisive-ish feedback, negative weights ------------
        self.i_to_e = b.SparseSynapses(self.n_inhib, self.n_cells)
        # One inhibitory spike must move a target's membrane potential by a
        # small fraction of threshold: ~1/30, since ~30 inhibitory cells are
        # co-active in the regime of interest.
        # Keep the *per-target* inhibitory convergence fixed instead of letting
        # it scale with the pool. At 4000 cells the pool is 800 wide, and a
        # fan_in of n_inhib/2 gave every excitatory cell 400 inhibitory
        # synapses against 200 at 2000 cells — measured, that halved the
        # completion score (0.43 recovered even from a complete cue).
        ie_fan_in = min(200, self.n_inhib)
        ie_scale = 0.033 * (self.n_inhib / 400.0)
        pre, post, w = b.build_random(
            self.n_inhib, self.n_cells, ie_fan_in, self.rng, ie_scale,
        )
        # gain folded into the weights: inhibition arrives as negative
        # current, and how negative is what balances the recurrence
        # strictly negative: a mixed-sign "inhibitory" pool is not inhibitory
        # at all (measured: with the sign bug, raising the gain from 2.4 to 12
        # *increased* activity instead of suppressing it)
        self.i_to_e.add_batch(
            pre, post, (-np.abs(w) * self.inh_gain).astype(np.float32)
        )

        self._prev_exc = np.empty(0, np.int64)
        self._prev_inh = np.empty(0, np.int64)
        self.i_syn = np.zeros(self.n_cells, np.float32)
        self.episodes: list[np.ndarray] = []

    # -- writing ---------------------------------------------------------

    def store(self, code) -> None:
        """Write one episode into the recurrent weights, in a single exposure.

        Every synapse between two cells of this episode is pushed toward
        ``w_max``. That is the biological write: co-active recurrent
        collaterals potentiate, and no repetition is required. Because only
        same-episode pairs are touched, later episodes barely disturb earlier
        ones unless they share cells — which is exactly the interference
        behaviour a real attractor has.
        """
        cells = np.nonzero(np.asarray(code, np.float32))[0]
        if cells.size == 0:
            return
        sids = self.rec.gather_out(cells)
        if sids.size:
            keep = np.isin(self.rec.post[sids], cells)
            sids = sids[keep]
            w = self.rec.w
            w[sids] = np.minimum(w[sids] + self.ltp, self.w_max)
        self.episodes.append(cells.astype(np.int32))

    @property
    def n_stored(self) -> int:
        return len(self.episodes)

    # -- reading ---------------------------------------------------------

    def _step_network(self, drive: np.ndarray):
        """One millisecond of CA3: recurrence, inhibition, two populations."""
        volley = np.zeros(self.n_cells, np.float32)
        if self._prev_exc.size:
            self.rec.propagate(self._prev_exc, volley)

        # integrate this volley into the decaying synaptic current
        self.i_syn *= self.syn_decay
        self.i_syn += volley

        current = self.i_syn.copy()

        # inhibitory feedback (strictly negative weights)
        inh_current = np.zeros(self.n_inhib, np.float32)
        if self._prev_exc.size:
            self.e_to_i.propagate(self._prev_exc, inh_current)
        if self._prev_inh.size:
            self.i_to_e.propagate(self._prev_inh, current)

        current += drive

        exc = self.exc.step(current)
        inh = self.inh.step(inh_current)
        self._prev_exc, self._prev_inh = exc, inh
        return exc, inh

    def reset(self) -> None:
        for pop in (self.exc, self.inh):
            if hasattr(pop, "v"):
                pop.v.fill(0.0)
                pop.adapt.fill(0.0)
                pop.resources.fill(1.0)
                pop.refractory.fill(0)
                pop._last_spikes = np.empty(0, np.int64)
        self._prev_exc = np.empty(0, np.int64)
        self._prev_inh = np.empty(0, np.int64)
        self.i_syn *= 0.0

    def recall(self, cue, steps: int = 30, settle: int = 10) -> np.ndarray:
        """Complete a partial cue back to a stored episode.

        The cue cells are driven for as long as it takes, then released; what
        the network does afterwards is the actual memory. A lookup table can be
        given a cue and answer instantly, so this distinction is the whole
        point — the answer here has to be produced by the recurrent weights
        settling, and it can fail.
        """
        cue_cells = np.nonzero(np.asarray(cue, np.float32))[0]
        self.reset()
        drive = np.zeros(self.n_cells, np.float32)
        if cue_cells.size:
            drive[cue_cells] = self.cue_current

        counts = np.zeros(self.n_cells, np.int32)
        total = max(steps, settle)
        for t in range(total):
            if t >= settle:
                drive = np.zeros(self.n_cells, np.float32)   # cue released
            exc, _ = self._step_network(drive)
            if t >= settle:
                counts[exc] += 1

        # Emit exactly as many cells as an episode contains. A relative
        # threshold ("fire if above 25% of the peak") lets the tail through —
        # measured, it returned 733 cells for a 120-cell episode, all correct
        # plus 600 hangers-on, which then diluted every downstream similarity
        # ratio to noise. Sparsity is part of the code, so it is enforced.
        out = np.zeros(self.n_cells, np.float32)
        live = np.nonzero(counts > 0)[0]
        if live.size == 0:
            return out
        if live.size > self.k:
            strength = counts[live].astype(np.float64)
            top = live[np.argsort(-strength, kind="stable")[: self.k]]
        else:
            top = live
        out[top] = 1.0
        return out

    # -- introspection ---------------------------------------------------

    def summary(self) -> str:
        cfg = (self.n_cells, self.fan_in, self.k, self.n_inhib)
        density = self.fan_in / max(1, self.n_cells)
        return (
            f"CA3Network(cells={cfg[0]}, fan_in={cfg[1]} "
            f"({density:.0%}), k={cfg[2]}, inhib={cfg[3]}, "
            f"same-episode inputs≈{density * self.k:.0f}, "
            f"episodes={self.n_stored})"
        )
