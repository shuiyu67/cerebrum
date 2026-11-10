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
        return idx

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
