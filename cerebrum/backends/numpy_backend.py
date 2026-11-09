"""Accelerated backend using NumPy.

Same interface as :mod:`cerebrum.backends.pure`, but propagation is done with
``np.add.at`` over the union of the presynaptic neurons' outgoing synapses, and
the per-neuron state lives in contiguous float arrays.

The outgoing index is rebuilt lazily: adding synapses marks it dirty and the
next propagation re-sorts. That keeps construction simple while keeping the
step loop vectorised.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class SparseSynapses:
    """Sparse synapse store with vectorised event-driven propagation."""

    __slots__ = ("n_pre", "n_post", "_pre", "_post", "_w", "_n", "_dirty",
                 "_order", "_indptr", "_in_order", "_in_indptr", "_in_dirty")

    def __init__(self, n_pre: int, n_post: int) -> None:
        self.n_pre = int(n_pre)
        self.n_post = int(n_post)
        self._pre = np.empty(0, np.int32)
        self._post = np.empty(0, np.int32)
        self._w = np.empty(0, np.float64)
        self._n = 0
        self._dirty = True
        self._order = np.empty(0, np.int64)
        self._indptr = np.zeros(self.n_pre + 1, np.int64)
        self._in_order = np.empty(0, np.int64)
        self._in_indptr = np.zeros(self.n_post + 1, np.int64)
        self._in_dirty = True

    # -- storage growth --------------------------------------------------

    def _ensure(self, extra: int) -> None:
        need = self._n + extra
        cap = self._pre.size
        if need <= cap:
            return
        new_cap = max(16, cap * 2, need)
        for name, dtype in (("_pre", np.int32), ("_post", np.int32), ("_w", np.float64)):
            old = getattr(self, name)
            grown = np.empty(new_cap, dtype)
            grown[: self._n] = old[: self._n]
            setattr(self, name, grown)

    def add(self, pre: int, post: int, weight: float) -> int:
        self._ensure(1)
        k = self._n
        self._pre[k] = pre
        self._post[k] = post
        self._w[k] = float(weight)
        self._n += 1
        self._dirty = True
        return k

    def add_batch(self, pres, posts, weights) -> None:
        pres = np.asarray(pres, np.int32)
        posts = np.asarray(posts, np.int32)
        ws = np.asarray(weights, np.float64)
        m = pres.size
        self._ensure(m)
        k = self._n
        self._pre[k : k + m] = pres
        self._post[k : k + m] = posts
        self._w[k : k + m] = ws
        self._n += m
        self._dirty = True

    def __len__(self) -> int:
        return int(self._n)

    # -- exported views --------------------------------------------------

    @property
    def pre(self) -> np.ndarray:
        return self._pre[: self._n]

    @property
    def post(self) -> np.ndarray:
        return self._post[: self._n]

    @property
    def w(self) -> np.ndarray:
        """Weights as a view: in-place edits are honoured by propagation."""
        return self._w[: self._n]

    # -- outgoing index --------------------------------------------------

    def _rebuild(self) -> None:
        pre = self._pre[: self._n]
        order = np.argsort(pre, kind="stable")
        sorted_pre = pre[order]
        indptr = np.zeros(self.n_pre + 1, np.int64)
        np.cumsum(np.bincount(sorted_pre, minlength=self.n_pre), out=indptr[1:])
        self._order = order
        self._indptr = indptr
        self._dirty = False

    def _rebuild_in(self) -> None:
        """Index synapses by their *target* neuron, for STDP's LTP term."""
        post = self._post[: self._n]
        order = np.argsort(post, kind="stable")
        sorted_post = post[order]
        indptr = np.zeros(self.n_post + 1, np.int64)
        np.cumsum(np.bincount(sorted_post, minlength=self.n_post), out=indptr[1:])
        self._in_order = order
        self._in_indptr = indptr
        self._in_dirty = False

    def gather_out(self, spikes):
        """Every synapse id whose presynaptic neuron is in ``spikes``."""
        if self._dirty:
            self._rebuild()
        if len(spikes) == 0:
            return np.empty(0, np.int64)
        idx = np.asarray(spikes, np.int64)
        lo = self._indptr[idx]
        counts = self._indptr[idx + 1] - lo
        total = int(counts.sum())
        if total == 0:
            return np.empty(0, np.int64)
        offsets = np.repeat(lo - np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
        return self._order[np.arange(total, dtype=np.int64) + offsets]

    def gather_in(self, spikes):
        """Every synapse id whose postsynaptic neuron is in ``spikes``."""
        if self._in_dirty:
            self._rebuild_in()
        if len(spikes) == 0:
            return np.empty(0, np.int64)
        idx = np.asarray(spikes, np.int64)
        lo = self._in_indptr[idx]
        counts = self._in_indptr[idx + 1] - lo
        total = int(counts.sum())
        if total == 0:
            return np.empty(0, np.int64)
        offsets = np.repeat(lo - np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
        return self._in_order[np.arange(total, dtype=np.int64) + offsets]

    def synapses_into_index(self, post: int):
        """Ids of every synapse arriving at ``post`` (indexed, O(1))."""
        if self._in_dirty:
            self._rebuild_in()
        return self._in_order[self._in_indptr[post] : self._in_indptr[post + 1]]

    def synapses_from(self, pre: int):
        if self._dirty:
            self._rebuild()
        return self._order[self._indptr[pre] : self._indptr[pre + 1]]

    # -- dynamics --------------------------------------------------------

    def _gather(self, spike_idx):
        """Synapse ids driven by the neurons in ``spike_idx``."""
        idx = np.asarray(spike_idx, np.int64)
        lo = self._indptr[idx]
        counts = self._indptr[idx + 1] - lo
        total = int(counts.sum())
        if total == 0:
            return None
        offsets = np.repeat(lo - np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
        return self._order[np.arange(total, dtype=np.int64) + offsets]

    def _scatter(self, target, values, current) -> None:
        """``current[target] += values``, choosing the cheaper NumPy path.

        ``np.add.at`` is unbuffered and costs several hundred nanoseconds per
        element, so it wins when a handful of synapses fire. ``bincount``
        allocates one array the size of the target population but streams, so
        it wins once tens of thousands of events arrive. The crossover sits
        around ``n_post / 8``; picking the wrong side costs 10x.
        """
        # Cost model, measured: np.add.at runs at ~150 ns per event, while
        # bincount streams at ~5 ns per event plus ~2 ns per slot. They cross
        # at total ~= n_post / 72. Using a factor of 8 here left a million-
        # neuron brain doing 100k unbuffered scatter writes per step: 259 ms
        # against 21 ms with the correct side of the branch.
        total = target.size
        if total * 72 < self.n_post:
            np.add.at(current, target, values)
        else:
            current += np.bincount(
                target, weights=values, minlength=self.n_post
            )[: self.n_post].astype(np.float32)

    def propagate(self, spike_idx, current) -> None:
        """``current[post] += w`` for every synapse whose presynaptic fired."""
        if self._dirty:
            self._rebuild()
        if len(spike_idx) == 0:
            return
        sids = self._gather(spike_idx)
        if sids is None:
            return
        self._scatter(self._post[sids], self._w[sids], current)

    def propagate_weighted(self, spike_idx, values, current) -> None:
        """``current[post] += w * value[pre]`` for the neurons that fired.

        Used by the sensor projection: each active sensor contributes its own
        intensity, not just presence.
        """
        if self._dirty:
            self._rebuild()
        if len(spike_idx) == 0:
            return
        idx = np.asarray(spike_idx, np.int64)
        lo = self._indptr[idx]
        counts = self._indptr[idx + 1] - lo
        total = int(counts.sum())
        if total == 0:
            return
        offsets = np.repeat(lo - np.concatenate(([0], np.cumsum(counts)[:-1])), counts)
        sids = self._order[np.arange(total, dtype=np.int64) + offsets]
        scale = np.repeat(np.asarray(values, np.float64), counts)
        self._scatter(self._post[sids],
                      (self._w[sids] * scale).astype(np.float32), current)

    # -- maintenance -----------------------------------------------------

    def synapses_into(self, post: int):
        return self.synapses_into_index(post)

    def compact(self, keep) -> int:
        """Drop synapses where ``keep`` is False; returns how many remain.

        Used by sleep pruning. Rebuilds both indexes.
        """
        keep = np.asarray(keep, bool)
        m = int(keep.sum())
        self._pre[:m] = self._pre[: self._n][keep]
        self._post[:m] = self._post[: self._n][keep]
        self._w[:m] = self._w[: self._n][keep]
        self._n = m
        self._dirty = True
        self._in_dirty = True
        return m

    def scale(self, factor: float) -> None:
        self._w[: self._n] *= factor

    def clip(self, limit: float) -> None:
        np.clip(self._w[: self._n], -limit, limit, out=self._w[: self._n])


class Rng:
    """Deterministic RNG backed by ``numpy.random.Generator``."""

    __slots__ = ("_r",)

    def __init__(self, seed: int | None = None) -> None:
        self._r = np.random.default_rng(seed)

    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        return float(self._r.normal(mu, sigma))

    def normal_list(self, n: int, mu: float = 0.0, sigma: float = 1.0) -> list[float]:
        return self._r.normal(mu, sigma, n).tolist()

    def uniform(self, lo: float = 0.0, hi: float = 1.0) -> float:
        return float(self._r.uniform(lo, hi))

    def randint(self, lo: int, hi: int) -> int:
        return int(self._r.integers(lo, hi))

    def choice(self, n: int, k: int) -> list[int]:
        return self._r.choice(n, size=k, replace=False).tolist()

    def random_list(self, n: int) -> list[float]:
        return self._r.random(n).tolist()


NAME = "numpy"
HAS_ACCELERATION = True


class Population:
    """A group of leaky integrate-and-fire neurons.

    State lives in contiguous float32 arrays and every update is a single
    vectorised expression, so the cost per step is memory bandwidth over
    ``n`` elements — around 0.09 ms at 10k neurons and 32 ms at 1M on this
    machine (see :mod:`cerebrum.bench.scale`).
    """

    __slots__ = ("n", "rng", "v", "threshold", "bias", "adapt", "refractory",
                 "resources", "_last_spikes",
                 "tau_m", "g_adapt", "tau_adapt", "refractory_steps",
                 "u_short_term", "tau_recovery", "target_hz",
                 "tau_homeostasis", "eta_homeostasis", "wta_k",
                 "region_sizes", "region_index", "gain_arr", "drive_arr",
                 "homeostasis_on", "wta_per_region", "region_starts")

    def __init__(self, n: int, rng: "Rng", *, tau_m: float = 10.0,
                 v_threshold: float = 1.0, refractory_steps: int = 3,
                 g_adapt: float = 0.3, tau_adapt: float = 150.0,
                 bias_mean: float = 0.13, bias_std: float = 0.19,
                 pacemaker_frac: float = 0.05, pacemaker_boost: float = 0.75,
                 u_short_term: float = 0.15, tau_recovery: float = 120.0,
                 target_hz: float = 5.0, tau_homeostasis: float = 2000.0,
                 eta_homeostasis: Optional[float] = None, wta_k: int = 0,
                 regions: Optional[tuple] = None,
                 wta_per_region: bool = False) -> None:
        self.n = int(n)
        self.rng = rng
        self.v = np.zeros(self.n, np.float32)
        self.threshold = np.ones(self.n, np.float32)
        self.bias = np.asarray(rng.normal_list(self.n, bias_mean, bias_std), np.float32)
        self.adapt = np.zeros(self.n, np.float32)
        self.refractory = np.zeros(self.n, np.int8)
        self.resources = np.ones(self.n, np.float32)
        self._last_spikes = np.empty(0, np.int64)

        n_pace = max(1, int(self.n * pacemaker_frac))
        if n_pace:
            self.bias[np.asarray(rng.choice(self.n, n_pace))] += pacemaker_boost

        self.tau_m = float(tau_m)
        self.g_adapt = float(g_adapt)
        self.tau_adapt = float(tau_adapt)
        self.refractory_steps = int(refractory_steps)
        self.u_short_term = float(u_short_term)
        self.tau_recovery = float(tau_recovery)
        self.target_hz = float(target_hz)
        self.tau_homeostasis = float(tau_homeostasis)
        # The rate of homeostasis follows from its time constant. It used to
        # be a hand-set 3e-5 while tau_homeostasis sat unused, which made the
        # loop 17x slower than declared: gain crawled from 1.00 to 1.04 over
        # 1600 steps, so a silent network needed many thousands of steps to
        # reach its target and looked "unable to self-stabilise".
        self.eta_homeostasis = (1.0 / max(1.0, float(tau_homeostasis))
                                if eta_homeostasis is None
                                else float(eta_homeostasis))
        # Competitive inhibition: at most wta_k neurons may fire per step.
        # Without it STDP has no winner to reinforce and weights saturate
        # instead of specialising.
        self.wta_k = int(wta_k)

        # Per-region homeostasis. One global gain cannot serve a population
        # whose regions receive different amounts of drive: measured on a
        # 400/1200/400 split, the global controller sat at its target while
        # the association and prediction regions were silent because the
        # sensory region was doing all the firing.
        self.region_sizes = tuple(int(x) for x in regions) if regions else (self.n,)
        if sum(self.region_sizes) != self.n:
            raise ValueError("regions must sum to the population size")
        self.region_index = np.repeat(
            np.arange(len(self.region_sizes)), self.region_sizes)
        self.region_starts = np.concatenate(
            ([0], np.cumsum(self.region_sizes)[:-1])).astype(np.int64)
        # Competition scoped to a local patch instead of the whole population.
        # Global k-WTA normalises everyone together, so no group can specialise
        # — every cell's activity is read against the same total. Cortical
        # inhibition is local (a basket cell covers a few hundred micrometres),
        # and that boundary is what lets neighbouring patches diverge.
        self.wta_per_region = bool(wta_per_region)
        self.gain_arr = np.ones(len(self.region_sizes), np.float64)
        self.drive_arr = np.zeros(len(self.region_sizes), np.float64)
        # Set False during sleep replay. Homeostasis scales the whole
        # population's excitability by a single factor, so it cancels out
        # whatever learning just did to the weights: measured, replaying 96
        # episodes changed representation separability by 0.01 against a
        # no-replay control, because the gain loop absorbed the difference.
        # Consolidation has to happen with the thermostat switched off.
        self.homeostasis_on = True

    # -- homeostasis state -----------------------------------------------

    @property
    def gain(self) -> float:
        return float(self.gain_arr.mean())

    @gain.setter
    def gain(self, value: float) -> None:
        self.gain_arr[:] = float(value)

    @property
    def drive(self) -> float:
        return float(self.drive_arr.mean())

    @drive.setter
    def drive(self, value: float) -> None:
        self.drive_arr[:] = float(value)

    # -- dynamics --------------------------------------------------------

    def step(self, current: np.ndarray, force=None) -> np.ndarray:
        """Advance one timestep, returning the indices that fired.

        ``force`` is a set of neuron indices compelled to fire regardless of
        their membrane potential. Sleep replay needs it: the hippocampus
        replays the *cortical pattern* — which cells fired together — not the
        sensory input, so the cells have to fire from the inside.
        """
        v = self.v
        v[:] = v + (-v + current * self.gain_arr[self.region_index]
                    + self.bias + self.drive_arr[self.region_index]
                    - self.g_adapt * self.adapt) / self.tau_m
        v[self.refractory > 0] = 0.0
        np.maximum(self.refractory - 1, 0, out=self.refractory)

        fired = v >= self.threshold
        if self.wta_k > 0:
            if self.wta_per_region and len(self.region_sizes) > 1:
                fired = self._wta_local(v, fired)
            else:
                n_fired = int(fired.sum())
                if n_fired > self.wta_k:
                    idx = np.nonzero(fired)[0]
                    top = idx[np.argpartition(v[idx], -self.wta_k)[-self.wta_k:]]
                    fired = np.zeros_like(fired)
                    fired[top] = True
        if force is not None and len(force):
            # applied after the competition, so a replayed pattern is not
            # thinned out by k-WTA
            forced = np.zeros(self.n, bool)
            forced[np.asarray(force, np.int64)] = True
            fired = fired | forced
        v[fired] = 0.0
        self.refractory[fired] = self.refractory_steps
        self.adapt += (fired.astype(np.float32) - self.adapt) * (1.0 / self.tau_adapt)
        self.resources -= self.u_short_term * fired
        np.clip(self.resources, 0.0, 1.0, out=self.resources)
        self.resources += (1.0 - self.resources) * (1.0 / self.tau_recovery)

        idx = np.nonzero(fired)[0]
        self._last_spikes = idx

        if not self.homeostasis_on:
            return idx

        # homeostasis, per region
        if idx.size:
            counts = np.bincount(self.region_index[idx],
                                 minlength=len(self.region_sizes))
        else:
            counts = np.zeros(len(self.region_sizes))
        rates = counts / np.asarray(self.region_sizes)
        error = self.target_hz - rates * 1000.0
        self.gain_arr *= np.exp(self.eta_homeostasis * error)
        np.clip(self.gain_arr, 0.05, 20.0, out=self.gain_arr)
        self.drive_arr += self.eta_homeostasis * 1.5 * error
        np.clip(self.drive_arr, -0.5, 1.8, out=self.drive_arr)
        return idx

    def _wta_local(self, v, fired):
        """k-WTA applied within each region separately.

        Each patch keeps its own ``k / n_regions`` winners, so patches compete
        with their own neighbours and not with the entire population.
        """
        n_reg = len(self.region_sizes)
        k_per = max(1, self.wta_k // n_reg)
        out = np.zeros_like(fired)
        for r in range(n_reg):
            lo = int(self.region_starts[r])
            hi = lo + int(self.region_sizes[r])
            seg = fired[lo:hi]
            n_seg = int(seg.sum())
            if n_seg > k_per:
                idx = np.nonzero(seg)[0]
                sub = v[lo:hi][idx]
                top = idx[np.argpartition(sub, -k_per)[-k_per:]]
                out[lo + top] = True
            else:
                out[lo:hi] = seg
        return out

    def rates(self) -> np.ndarray:
        """Placeholder for a running rate estimate (single-step indicator)."""
        out = np.zeros(self.n, np.float32)
        out[self._last_spikes] = 1.0
        return out


def zeros(n: int):
    """A fresh zero-filled current buffer."""
    return np.zeros(n, np.float32)


def group_counts(spike_idx, n_groups: int, group_size: int):
    """Number of spikes per output group, as a flat list."""
    if len(spike_idx) == 0:
        return [0.0] * n_groups
    groups = np.asarray(spike_idx, np.int64) // group_size
    counts = np.bincount(groups, minlength=n_groups)[:n_groups]
    return counts.astype(np.float64).tolist()


def build_random(n_pre: int, n_post: int, fan_in: int, rng: "Rng",
                 scale: float = 1.0):
    """Vectorised random projection ``(n_pre -> n_post)``.

    Returns ``(pre, post, weight)`` arrays ready for
    :meth:`SparseSynapses.add_batch`. Construction cost is one call rather
    than ``n_post * fan_in`` Python-level iterations, which is what makes a
    million-neuron brain build in under a second.
    """
    total = n_post * fan_in
    pre = rng._r.integers(0, n_pre, size=total, dtype=np.int32)  # noqa: SLF001
    post = np.repeat(np.arange(n_post, dtype=np.int32), fan_in)
    weight = rng._r.normal(0.0, scale, total).astype(np.float32)  # noqa: SLF001
    return pre, post, weight


def build_sensor(n_inputs: int, n_neurons: int, span: int, rng: "Rng",
                 scale: float = 1.0):
    """Sparse sensor projection with a bounded receptive field.

    Each sensor drives ``span`` neurons, so an input vector of width ``W``
    touches at most ``W * span`` neurons. Without this bound a sensor vector
    drives the whole population and the step cost collapses: at 128 sensors
    over 1M neurons, the naive ``fan_in`` layout produced 4M synaptic events
    per step (214 ms), against 12.8k with ``span=100`` (2 ms).
    """
    total = n_inputs * span
    pre = np.repeat(np.arange(n_inputs, dtype=np.int32), span)
    post = rng._r.integers(0, n_neurons, size=total, dtype=np.int32)  # noqa: SLF001
    weight = rng._r.normal(0.0, scale, total).astype(np.float32)  # noqa: SLF001
    return pre, post, weight


def add_block(dst, src) -> None:
    """``dst[:len(src)] += src`` — used when a projection targets one region."""
    dst[: len(src)] += src
