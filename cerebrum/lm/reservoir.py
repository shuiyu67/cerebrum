"""A million-neuron reservoir that can be trained on a language modelling task.

Why a reservoir and not a trained recurrent network
---------------------------------------------------
This engine has no backward pass and never will — the point of the project is
that every mechanism changes a synaptic weight locally. So "train an LM" has
to mean something a gradient-free network can actually do.

A reservoir is the honest answer. The million neurons are a fixed, random,
recurrently coupled pool. They are not trained at all. What is trained is a
single linear readout on top, and it is trained by plain softmax
cross-entropy. That is one layer of gradient, not backpropagation through a
network, and it is the same amount of gradient a logistic regression uses.

What that buys: the recurrent pool turns the last K characters into a
high-dimensional nonlinear state, and the readout learns P(next | state). It
will not beat a transformer. It should beat a bigram table, and if it does
not, the reservoir is contributing nothing and the number says so.

Scaling
-------
The assembled brain in :mod:`cerebrum.brain` uses dense ``n x n`` matrices per
region, which is correct at 900 cells and impossible at a million
(10^12 floats, four terabytes). This module uses a different representation:

* synapses are **grouped by presynaptic cell** — each neuron owns exactly
  ``fan_out`` contiguous outgoing synapses, so an index range is a slice and
  no sorting is ever needed;
* propagation is one ``bincount`` over the events of the cells that fired;
* there is no per-cell threshold adaptation, only adaptation and a global
  inhibitory pool.

At 10^6 neurons and ``fan_out = 16`` that is 16M synapses in 128 MB, and a
step costs about 10 ms.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class SparseReservoir:
    """A fixed random recurrent pool of leaky integrate-and-fire neurons.

    Parameters
    ----------
    n:
        Number of neurons.
    fan_out:
        Outgoing synapses per neuron. Total synapses are ``n * fan_out``.
    tau_m:
        Membrane time constant in milliseconds.
    w_scale:
        Standard deviation of the initial synaptic weights. This is the knob
        that decides whether the pool is quiet, critical or epileptic, and at
        1M cells there is very little room between the last two.

        It has to be this large because the membrane update multiplies the
        synaptic current by ``1/tau``: at the physiologically sensible
        ``w = 0.1`` a single input contributes 0.005 to the membrane, so
        thirty of them have to arrive in the same millisecond to matter. The
        pool was, in effect, disconnected — measured echo after the input
        stopped was exactly zero at every fan-out tried, up to 128. At
        ``w = 8`` the same pool reverberates for 5-10 ms and settles at a
        stable 10 Hz.
    ginh:
        Strength of the global inhibitory pool, applied in proportion to how
        much of the population fired last step. Without it a random recurrent
        pool with any interesting gain runs away within a few hundred steps.
    """

    def __init__(
        self,
        n: int = 1_000_000,
        fan_out: int = 32,
        *,
        tau_m: float = 20.0,
        v_threshold: float = 1.0,
        refractory: int = 2,
        g_adapt: float = 0.30,
        tau_adapt: float = 150.0,
        tau_adapt_spread: float = 0.0,
        bias: float = 0.06,
        w_scale: float = 8.0,
        ginh: float = 30.0,
        target_frac: float = 0.03,
        dale: bool = False,
        inh_frac: float = 0.2,
        normalize_out: bool = False,
        regular: bool = False,
        plasticity: float = 0.0,
        target_rate_hz: float = 10.0,
        plasticity_every: int = 10,
        stdp: float = 0.0,
        tau_trace: float = 20.0,
        a_plus: float = 1.0,
        a_minus: float = 1.05,
        seed: int = 0,
    ) -> None:
        self.n = int(n)
        self.fan_out = int(fan_out)
        self.tau_m = float(tau_m)
        self.v_th = float(v_threshold)
        self.refractory = int(refractory)
        self.g_adapt = float(g_adapt)
        self.tau_adapt = float(tau_adapt)
        # Heterogeneous adaptation. Cortical neurons do not share one
        # adaptation time constant — they span tens to hundreds of
        # milliseconds, and that spread is a bank of leaky integrators with
        # different horizons. One constant makes every cell forget at the same
        # rate, which is exactly a single-pole low-pass filter; a spread
        # encodes a weighted history over a range of scales in the population
        # state, which is linear-readable memory that does not require the
        # recurrence to be non-chaotic.
        self.tau_adapt_spread = float(tau_adapt_spread)
        self.ginh = float(ginh)
        self.target_frac = float(target_frac)
        self.beta = 1.0 / self.tau_m

        rng = np.random.default_rng(seed)
        # Synapses grouped by presynaptic cell: cell i owns the slice
        # [i*fan_out, (i+1)*fan_out). This is what makes an index range a
        # slice instead of a search.
        total = self.n * self.fan_out
        self.post = rng.integers(0, self.n, total, dtype=np.int32)

        # --- in-degree regularity ---------------------------------------
        # The default draw is an Erdos-Renyi graph, so in-degree is Poisson:
        # mean `fan_out`, but individual cells receive anywhere from 30 to 100+
        # synapses. The cells at the top of that tail fire first and drive
        # their own neighbours hard, which is a local instability that a
        # *global* statistic cannot see. Measured: the pool holds a healthy
        # 10-20 Hz mean rate while the largest Lyapunov exponent is +0.05,
        # and every inhibitory gain from 5 to 160 leaves the exponent positive.
        #
        # `regular=True` builds a directed random-regular graph instead: each
        # cell sends exactly `fan_out` synapses and receives exactly
        # `fan_out`, by summing `fan_out` random permutations. Each
        # permutation is a bijection, so every target is hit exactly once per
        # block and the in-degree has zero variance. Same memory, same
        # synapse count, same mean strength — only the tail is gone.
        self.regular = bool(regular)
        if self.regular:
            perms = np.empty((self.fan_out, self.n), dtype=np.int32)
            for j in range(self.fan_out):
                perms[j] = rng.permutation(self.n)
            # Cell i owns slice [i*fan_out, (i+1)*fan_out), so the flat layout
            # has to be (cell, slot), not (slot, cell).
            self.post = np.ascontiguousarray(perms.T).reshape(-1)
            del perms
        self.w = rng.normal(0.0, w_scale, total).astype(np.float32)

        # --- Dale's law -------------------------------------------------
        # The shipped pool draws every synapse's sign independently, so one
        # cell emits both excitation and inhibition. Cortex does not do this:
        # a neuron is excitatory or inhibitory for its whole life, and about
        # one in five is inhibitory. The distinction is not cosmetic here.
        # Feedback inhibition is what stops a local cluster of strongly
        # coupled excitatory cells from running away, and with mixed signs a
        # cell cannot be suppressed by its own neighbours — only nudged. That
        # is a candidate explanation for why the measured Lyapunov exponent
        # stays positive while the mean firing rate looks healthy: the mean is
        # fine, the variance around it is not.
        self.dale = bool(dale)
        self.inh_mask = None
        if self.dale:
            n_inh = max(1, int(round(self.n * float(inh_frac))))
            inh_cells = rng.choice(self.n, n_inh, replace=False)
            mask = np.zeros(self.n, dtype=bool)
            mask[inh_cells] = True
            cell_sign = np.where(mask, np.float32(-1.0),
                                 np.float32(1.0)).astype(np.float32)
            self.w = (np.abs(self.w)
                      * np.repeat(cell_sign, self.fan_out)).astype(np.float32)
            self.inh_mask = mask

        # --- synaptic scaling -------------------------------------------
        # Hold each cell's total outgoing strength fixed instead of letting a
        # normal draw decide it. A Gaussian with sd = w_scale gives cell-to-
        # cell output norms a spread of roughly 1/sqrt(2*fan_out), so at
        # fan_out 32 the strongest cells carry about 12% more drive than the
        # mean cell. If the chaos threshold sits inside that spread, the
        # population reports a healthy mean rate while its top decile is
        # already diverging — which is exactly the pattern measured. Scaling
        # removes the spread and leaves only the mean to tune.
        self.normalize_out = bool(normalize_out)
        if self.normalize_out:
            wm = self.w.reshape(self.n, self.fan_out)
            nrm = np.linalg.norm(wm, axis=1, keepdims=True)
            nrm[nrm < 1e-9] = 1e-9
            wm *= (np.float32(w_scale * np.sqrt(self.fan_out)) / nrm)
        self._offsets = (np.arange(self.fan_out, dtype=np.int64)
                         * np.float32(1.0))
        self._arange = np.arange(self.fan_out, dtype=np.int64)

        # Random initial membrane potentials break the symmetry that would
        # otherwise make every driven cell fire on the same millisecond: a
        # perfectly synchronous input volley arrives at each target as one
        # coincident event and the pool never develops a spread of
        # latencies, which is where a reservoir's memory lives.
        self.v = rng.normal(0.0, 0.25, self.n).astype(np.float32)

        # Readout projection: a fixed sparse *signed* random matrix, four
        # non-zeros per neuron. Folding spikes into bins with `idx % n_bins`
        # is an unsigned count — every neuron adds to one bucket and the sign
        # information that separates contexts is thrown away. Measured, that
        # projection left the model worse than a bigram table fitted on the
        # same 8,000 characters (4.83 vs 4.44 bits).
        proj_fan = 4
        self.n_bins = 0
        self.proj_post = rng.integers(0, self.n, self.n * proj_fan,
                                      dtype=np.int32)
        self.proj_val = rng.choice([-1.0, 1.0], self.n * proj_fan
                                   ).astype(np.float32)
        self.proj_fan = proj_fan
        self.adapt = np.zeros(self.n, np.float32)
        self.refr = np.zeros(self.n, np.int8)
        self.bias = np.float32(bias)
        self.prev = np.empty(0, np.int64)
        self.spikes_total = 0
        self.steps_total = 0

        # --- plasticity -------------------------------------------------
        # Everything above freezes the pool at initialisation. That is the
        # reason the readout is the only thing that ever learns, and the
        # reason the pool is worth 0.05 bits: a fixed random projection
        # carries a fixed amount of information, and no amount of training on
        # top of it can exceed that.
        #
        # The first rule here is synaptic scaling (Turrigiano 1998), and it is
        # deliberately the most local rule available: a cell compares its own
        # recent firing rate against a target and multiplies its own outgoing
        # synapses by a factor. No reverse index, no error signal, no gradient,
        # nothing global. One float per neuron is the entire state.
        #
        # It attacks the actual failure mode rather than a symptom. Chaos here
        # starts with a cell that fires slightly more than its neighbours,
        # receives more input because of it, and fires more still. Scaling
        # cuts that loop at the cell: the ones running hot turn their own
        # output down, so the divergence has nowhere to build. If that is
        # right, the pool should walk itself to lambda ~ 0 without anyone
        # tuning w or ginh.
        self.plasticity = float(plasticity)
        self.target_rate = float(target_rate_hz) / 1000.0
        self.plasticity_every = max(1, int(plasticity_every))
        self.rate_ema = np.zeros(self.n, np.float32)
        self._rate_decay = np.float32(np.exp(-1.0 / 100.0))
        self._rate_step = np.float32(1.0 - self._rate_decay)
        self.w_max = np.float32(max(1e-6, abs(float(w_scale)) * 4.0))
        self.plasticity_updates = 0

        # Scaling has to act on *incoming* synapses. A cell that fires too
        # much must weaken what it receives; acting on the outgoing block
        # controls the neighbours, not the cell. Measured the wrong way round:
        # 23 Hz -> 135 Hz within 1500 steps, output norm 23.9 -> 86.9 (the
        # clip ceiling times the fan-out), capacity 0.80 -> 0.13. A silent
        # cell had its outgoing weights amplified to the maximum and then
        # fired a volley into its targets.
        #
        # Post-synaptic grouping is a one-off sort; the synapses never move,
        # so this index is built once and reused for every update.
        self._order = None
        self._post_sorted = None
        if self.plasticity > 0.0:
            self._order = np.argsort(self.post, kind="stable")
            self._post_sorted = self.post[self._order].astype(np.int64)

        # --- STDP -------------------------------------------------------
        # Scaling only touches amplitudes, and amplitude is the knob that sets
        # the Lyapunov exponent, which is the knob that sets capacity.
        # Measured: per-cell scaling drives the incoming-norm spread from
        # cv 0.11 to 0.87 and pushes lambda from +0.083 to +0.221. It cannot
        # escape the lambda/capacity trade-off because it *is* that trade-off,
        # automated.
        #
        # Spike-timing dependent plasticity changes a different thing: which
        # path is strong, not how strong the cell is overall. Paired with a
        # hard incoming-norm constraint (Song & Abbott 2000), the total input
        # per cell is held fixed, so the population gain stays put while the
        # distribution is free to specialise. If capacity rises at a fixed
        # exponent, the information lives in the weights rather than in the
        # gain - which is the only way the ceiling measured here can move.
        self.stdp = float(stdp)
        self._a_plus = float(a_plus)
        self._a_minus = float(a_minus)
        self.tau_trace = float(tau_trace)
        self._trace_decay = np.float32(np.exp(-1.0 / self.tau_trace))
        self.pre_trace = np.zeros(self.n, np.float32)
        self.post_trace = np.zeros(self.n, np.float32)
        self._pre_sorted = None
        self._target_in_norm = float(abs(w_scale)) * np.sqrt(self.fan_out)
        self.stdp_updates = 0
        if self.stdp > 0.0:
            if self._order is None:
                self._order = np.argsort(self.post, kind="stable")
                self._post_sorted = self.post[self._order].astype(np.int64)
            pre_index = np.repeat(np.arange(self.n, dtype=np.int32), self.fan_out)
            self._pre_sorted = pre_index[self._order]

        # Preallocated temporaries and folded constants for the in-place
        # update. Allocating these per step was most of the runtime.
        self._inbuf = np.zeros(self.n, np.float32)
        self._tmp = np.zeros(self.n, np.float32)
        self._one_minus_beta = np.float32(1.0 - self.beta)
        self._bias_beta = np.float32(self.bias * self.beta)
        self._k_adapt = np.float32(self.g_adapt * self.beta)
        if self.tau_adapt_spread > 0.0:
            lo = self.tau_adapt / (1.0 + self.tau_adapt_spread)
            hi = self.tau_adapt * (1.0 + self.tau_adapt_spread)
            self.tau_adapt_arr = np.exp(
                rng.uniform(np.log(lo), np.log(hi), self.n)).astype(np.float32)
            self._adapt_decay = (1.0 - 1.0 / self.tau_adapt_arr).astype(np.float32)
        else:
            self.tau_adapt_arr = None
            self._adapt_decay = np.float32(1.0 - 1.0 / self.tau_adapt)

    # -- dynamics --------------------------------------------------------

    def step(self, ext_idx: Optional[np.ndarray] = None,
             ext_val: float = 3.0) -> np.ndarray:
        """Advance one millisecond. Returns the indices that fired.

        Every arithmetic step is in-place over preallocated arrays. The
        straightforward version — allocating a temporary per operation — cost
        32 ms per step at a million neurons, of which the actual synaptic
        arithmetic was under two. The rest was memory bandwidth spent
        creating and discarding 4 MB arrays.
        """
        n = self.n
        prev = self.prev
        v = self.v

        # --- synaptic current -------------------------------------------
        if prev.size:
            sel = (prev[:, None] * self.fan_out + self._arange[None, :]).ravel()
            post = self.post[sel]
            wts = self.w[sel]
            # Unbuffered scatter beats bincount when there are few events:
            # bincount always allocates and zeroes `n` slots, which at 1M
            # neurons is 8 MB for nothing. The two cross around n/72 events.
            if sel.size * 72 < n:
                cur = self._inbuf
                cur.fill(0.0)
                np.add.at(cur, post, wts)
            else:
                cur = np.bincount(post, weights=wts,
                                  minlength=n)[:n].astype(np.float32)

            excess = prev.size / n - self.target_frac
            if excess > 0.0:
                cur -= np.float32(self.ginh * excess)

            # --- membrane, fused ----------------------------------------
            v *= self._one_minus_beta
            np.multiply(cur, self.beta, out=cur)
            v += cur
        else:
            v *= self._one_minus_beta

        v += self._bias_beta
        np.multiply(self.adapt, self._k_adapt, out=self._tmp)
        v -= self._tmp

        if ext_idx is not None and len(ext_idx):
            # a current, not a current divided by tau: the input has to be
            # able to fire a cell on the step it arrives.
            #
            # A vector is allowed so the same population can be driven with
            # cell-specific gains. Uniform drive locks the driven cells into
            # phase — they charge together, cross threshold together, and go
            # refractory together — so the firing count reflects the
            # refractory phase rather than the input. Measured with uniform
            # drive: correlation between drive magnitude and spike count 0.48,
            # and no single readout dimension above 0.31, which made the
            # memory-capacity measurement read as zero for every pool.
            ei = np.asarray(ext_idx, np.int64)
            if np.ndim(ext_val) == 0:
                v[ei] += np.float32(ext_val)
            else:
                v[ei] += np.asarray(ext_val, np.float32)

        # --- refractoriness and threshold -------------------------------
        # The refractory clamp is folded into the threshold test instead of
        # being a separate masked write over the whole population. A boolean
        # fancy-assignment on 1M elements costs as much as the arithmetic
        # around it, and it was clamping cells that could not have fired
        # anyway.
        np.maximum(self.refr - 1, 0, out=self.refr)
        fired = (v >= self.v_th) & (self.refr == 0)
        idx = np.nonzero(fired)[0]
        if idx.size:
            v[idx] = 0.0
            self.refr[idx] = self.refractory
            self.adapt[idx] += np.float32(1.0)

        np.multiply(self.adapt, self._adapt_decay, out=self.adapt)

        self.prev = idx
        self.spikes_total += int(idx.size)
        self.steps_total += 1

        if self.plasticity > 0.0:
            # Exponentially weighted firing rate, ~100 ms. A cell that fires
            # once steps its estimate up by one minus the decay, which is the
            # standard way to read a rate out of spikes without a counter.
            self.rate_ema *= self._rate_decay
            if idx.size:
                self.rate_ema[idx] += self._rate_step
            if self.steps_total % self.plasticity_every == 0:
                self._scale_synapses()

        if self.stdp > 0.0:
            self.pre_trace *= self._trace_decay
            self.post_trace *= self._trace_decay
            if idx.size:
                self.pre_trace[idx] += 1.0
                self.post_trace[idx] += 1.0
                self._stdp_update(idx)
            if self.steps_total % self.plasticity_every == 0:
                self._normalize_incoming()

        return idx

    def _stdp_update(self, idx: np.ndarray) -> None:
        """The classic pair rule, applied as two independent vectorised passes.

        Depression first, on the outgoing block of the cells that just fired:
        each of their synapses is weakened by how recently its target fired.
        Those blocks are contiguous in the shipped layout, so this pass is one
        gather, one subtract and one scatter.

        Potentiation second, on the incoming synapses of the same cells: each
        is strengthened by how recently its source fired. In-degree grouping
        comes from a precomputed sort, so the slice boundaries are a
        ``searchsorted`` rather than a scan.

        Order matters — depression writes to ``self.w`` directly, and
        potentiation copies *after* that, so the second pass sees the first
        one's result. Doing it the other way round silently discards the
        depression.
        """
        lr = self.stdp

        # --- depression (original layout, contiguous per presynaptic cell) --
        sel = (idx[:, None] * self.fan_out + self._arange[None, :]).ravel()
        if sel.size:
            ws = self.w[sel]
            ws -= lr * self._a_minus * self.post_trace[self.post[sel]]
            np.clip(ws, -self.w_max, self.w_max, out=ws)
            self.w[sel] = ws

        # --- potentiation (in-degree order, gathered after depression) ------
        order = self._order
        wt = self.w[order]
        starts = np.searchsorted(self._post_sorted, idx, side="left")
        ends = np.searchsorted(self._post_sorted, idx, side="right")
        for s, e in zip(starts, ends):
            if e > s:
                sl = slice(int(s), int(e))
                wt[sl] += lr * self._a_plus * self.pre_trace[self._pre_sorted[sl]]
        np.clip(wt, -self.w_max, self.w_max, out=wt)
        self.w[order] = wt
        self.stdp_updates += 1

    def _normalize_incoming(self) -> None:
        """Hold every cell's incoming weight norm fixed.

        This is the stabiliser, and it is what makes the rule interesting:
        STDP decides the *distribution* of strengths onto a cell, this decides
        the *total*. Without it, potentiation grows without bound and the pool
        runs away; with it, the only thing that can change is which inputs
        dominate, which is exactly the quantity that could carry a sequence.
        """
        sq = np.bincount(self.post, weights=self.w.astype(np.float64) ** 2,
                         minlength=self.n)
        cur = np.sqrt(sq)
        cur[cur < 1e-9] = 1e-9
        factor = (self._target_in_norm / cur).astype(np.float32)
        self.w *= factor[self.post]

    def _scale_synapses(self) -> None:
        """Scale each cell's *incoming* synapses by its own rate error.

        Synaptic scaling in the sense of Turrigiano: a cell that has been
        firing above its target weakens everything it receives, a cell below
        target strengthens. The update uses one scalar per cell and touches
        only that cell's own incoming synapses, so it is local by
        construction: no error signal, no reverse pass, no gradient. Cost is
        one gather, one multiply and one scatter over the synapses.

        The clip is not cosmetic. Without a bound the rule has a fixed point
        only at zero: cells above target shrink until the pool dies and cells
        below target grow until it saturates. Bounding at +-4x the initial
        scale keeps both failure modes finite while still allowing a 4x move,
        which is much larger than the ~12% spread that separates a stable cell
        from a chaotic one.
        """
        err = (self.target_rate - self.rate_ema) / self.target_rate
        factor = 1.0 + self.plasticity * err
        np.clip(factor, 0.5, 2.0, out=factor)
        f_all = factor[self._post_sorted]
        ws = self.w[self._order]
        ws *= f_all
        np.clip(ws, -self.w_max, self.w_max, out=ws)
        self.w[self._order] = ws
        self.plasticity_updates += 1

    def weight_stats(self) -> dict:
        """Incoming-weight spread. The rule works on in-degree, so the spread
        it is supposed to remove has to be measured on the same axis."""
        sq = np.bincount(self.post, weights=self.w.astype(np.float64) ** 2,
                         minlength=self.n)
        norms = np.sqrt(sq)
        return {"mean": float(norms.mean()), "sd": float(norms.std()),
                "cv": float(norms.std() / (norms.mean() + 1e-12)),
                "updates": self.plasticity_updates,
                "rate_hz": self.rate_hz()}

    # -- state readout ---------------------------------------------------

    def binned_state(self, idx: np.ndarray, n_bins: int = 4096) -> np.ndarray:
        """Signed random projection of the active cells onto ``n_bins``.

        The projection matrix is fixed and sparse (four non-zeros per cell),
        with ±1 weights, so two contexts that activate overlapping but
        different cell sets land in different directions rather than in the
        same count.
        """
        out = np.zeros(n_bins, np.float32)
        idx = np.asarray(idx, np.int64)
        if idx.size == 0:
            return out
        sel = (idx[:, None] * self.proj_fan
               + np.arange(self.proj_fan)[None, :]).ravel()
        post = self.proj_post[sel] % n_bins
        return np.bincount(post, weights=self.proj_val[sel],
                           minlength=n_bins)[:n_bins].astype(np.float32)

    def reset(self) -> None:
        self.v.fill(0.0)
        self.adapt.fill(0.0)
        self.refr.fill(0)
        self.prev = np.empty(0, np.int64)

    def rate_hz(self) -> float:
        if self.steps_total == 0:
            return 0.0
        return self.spikes_total / self.steps_total / self.n * 1000.0

    def memory_mb(self) -> float:
        total = self.post.nbytes + self.w.nbytes
        state = self.v.nbytes + self.adapt.nbytes + self.refr.nbytes
        return (total + state) / 2 ** 20

    def summary(self) -> str:
        return (f"SparseReservoir(n={self.n:,}, fan_out={self.fan_out}, "
                f"synapses={self.n * self.fan_out:,}, "
                f"{self.memory_mb():.0f} MB, {self.rate_hz():.1f} Hz)")
