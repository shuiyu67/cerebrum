"""Zero-dependency backend: standard library only.

Storage is three parallel lists (pre, post, weight) plus a reverse index from
each presynaptic neuron to the synapses it drives. Propagation is *event
driven*: only the synapses whose source fired are touched, so a step costs
``O(spikes × fan_out)`` instead of ``O(synapses)``. That is what makes a
2000-neuron / 40k-synapse network run at roughly half a millisecond per step
in pure Python.

Everything here uses only ``random`` and basic types, so the engine runs on a
bare Python install.
"""

from __future__ import annotations

import random


class SparseSynapses:
    """Event-driven sparse synapse store.

    Parameters
    ----------
    n_pre, n_post:
        Sizes of the presynaptic and postsynaptic populations.
    """

    __slots__ = ("n_pre", "n_post", "pre", "post", "w", "_out")

    def __init__(self, n_pre: int, n_post: int) -> None:
        self.n_pre = int(n_pre)
        self.n_post = int(n_post)
        self.pre: list[int] = []
        self.post: list[int] = []
        self.w: list[float] = []
        self._out: list[list[int]] = [[] for _ in range(self.n_pre)]

    # -- construction ----------------------------------------------------

    def add(self, pre: int, post: int, weight: float) -> int:
        """Append one synapse, returning its id."""
        sid = len(self.w)
        self.pre.append(pre)
        self.post.append(post)
        self.w.append(float(weight))
        self._out[pre].append(sid)
        return sid

    def add_batch(self, pres, posts, weights) -> None:
        """Append many synapses at once."""
        for p, q, w in zip(pres, posts, weights):
            self.add(p, q, w)

    def __len__(self) -> int:
        return len(self.w)

    # -- dynamics --------------------------------------------------------

    def propagate(self, spike_idx, current) -> None:
        """Add synaptic input to ``current`` in place.

        ``current[k]`` receives ``w`` for every synapse whose presynaptic
        neuron appears in ``spike_idx``. Only those synapses are visited.
        """
        post = self.post
        w = self.w
        out = self._out
        for pre in spike_idx:
            for sid in out[pre]:
                current[post[sid]] += w[sid]

    def propagate_weighted(self, spike_idx, values, current) -> None:
        """``current[post] += w * value[pre]`` for the neurons that fired."""
        post = self.post
        w = self.w
        out = self._out
        for pre, val in zip(spike_idx, values):
            if val == 0.0:
                continue
            for sid in out[pre]:
                current[post[sid]] += w[sid] * val

    # -- plasticity helpers ----------------------------------------------

    def synapses_from(self, pre: int):
        """Ids of every synapse driven by ``pre``."""
        return self._out[pre]

    def synapses_into(self, post: int):
        """Ids of every synapse arriving at ``post``.

        Linear scan; only used by offline maintenance (pruning, normalisation),
        never inside the step loop.
        """
        return [sid for sid, q in enumerate(self.post) if q == post]

    def scale(self, factor: float) -> None:
        self.w = [x * factor for x in self.w]

    def clip(self, limit: float) -> None:
        lim = float(limit)
        self.w = [lim if x > lim else (-lim if x < -lim else x) for x in self.w]


class Rng:
    """Deterministic RNG with the small surface the engine needs."""

    __slots__ = ("_r",)

    def __init__(self, seed: int | None = None) -> None:
        self._r = random.Random(seed)

    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        return self._r.gauss(mu, sigma)

    def normal_list(self, n: int, mu: float = 0.0, sigma: float = 1.0) -> list[float]:
        g = self._r.gauss
        return [g(mu, sigma) for _ in range(n)]

    def uniform(self, lo: float = 0.0, hi: float = 1.0) -> float:
        return self._r.uniform(lo, hi)

    def randint(self, lo: int, hi: int) -> int:
        """Half-open ``[lo, hi)``."""
        return self._r.randrange(lo, hi)

    def choice(self, n: int, k: int) -> list[int]:
        """``k`` distinct indices from ``range(n)``."""
        return self._r.sample(range(n), k)

    def random_list(self, n: int) -> list[float]:
        r = self._r.random
        return [r() for _ in range(n)]


NAME = "pure"
HAS_ACCELERATION = False


class Population:
    """A group of leaky integrate-and-fire neurons (standard library only).

    Every update is an explicit loop over Python floats. That is roughly two
    orders of magnitude slower per neuron than the NumPy backend, so this
    backend is meant for small networks (a few thousand neurons) or for
    machines without NumPy, not for the million-neuron regime.
    """

    __slots__ = ("n", "rng", "v", "threshold", "bias", "adapt", "refractory",
                 "resources", "_last_spikes", "gain", "drive",
                 "tau_m", "g_adapt", "tau_adapt", "refractory_steps",
                 "u_short_term", "tau_recovery", "target_hz",
                 "tau_homeostasis", "eta_homeostasis", "wta_k",
                 "pacemaker_boost")

    def __init__(self, n: int, rng: "Rng", *, tau_m: float = 10.0,
                 v_threshold: float = 1.0, refractory_steps: int = 3,
                 g_adapt: float = 0.3, tau_adapt: float = 150.0,
                 bias_mean: float = 0.13, bias_std: float = 0.19,
                 pacemaker_frac: float = 0.05, pacemaker_boost: float = 0.75,
                 u_short_term: float = 0.15, tau_recovery: float = 120.0,
                 target_hz: float = 5.0, tau_homeostasis: float = 2000.0,
                 eta_homeostasis: Optional[float] = None, wta_k: int = 0) -> None:
        self.n = int(n)
        self.rng = rng
        self.v = [0.0] * self.n
        self.threshold = [float(v_threshold)] * self.n
        self.bias = rng.normal_list(self.n, bias_mean, bias_std)
        self.adapt = [0.0] * self.n
        self.refractory = [0] * self.n
        self.resources = [1.0] * self.n
        self._last_spikes: list[int] = []
        self.gain = 1.0
        self.drive = 0.0
        self.pacemaker_boost = float(pacemaker_boost)

        n_pace = max(1, int(self.n * pacemaker_frac))
        for k in rng.choice(self.n, n_pace):
            self.bias[k] += self.pacemaker_boost

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
        self.wta_k = int(wta_k)

    def step(self, current, force=None) -> list[int]:
        """Advance one timestep, returning the indices that fired.

        ``force`` compels the listed neurons to fire regardless of membrane
        potential — used by sleep replay, which re-activates the cortical
        pattern rather than the sensory input.
        """
        v = self.v
        bias = self.bias
        adapt = self.adapt
        refr = self.refractory
        res = self.resources
        thr = self.threshold
        inv_tau = 1.0 / self.tau_m
        gain = self.gain
        drive = self.drive
        g_adapt = self.g_adapt
        inv_adapt = 1.0 / self.tau_adapt
        inv_rec = 1.0 / self.tau_recovery
        u = self.u_short_term
        refr_steps = self.refractory_steps

        fired: list[int] = []
        for k in range(self.n):
            if refr[k] > 0:
                refr[k] -= 1
                v[k] = 0.0
                continue
            v[k] += (-v[k] + current[k] * gain + bias[k] + drive
                     - g_adapt * adapt[k]) * inv_tau
            if v[k] >= thr[k]:
                v[k] = 0.0
                refr[k] = refr_steps
                adapt[k] += (1.0 - adapt[k]) * inv_adapt
                res[k] = max(0.0, res[k] - u) + (1.0 - max(0.0, res[k] - u)) * inv_rec
                fired.append(k)
            else:
                adapt[k] -= adapt[k] * inv_adapt
                res[k] += (1.0 - res[k]) * inv_rec

        if self.wta_k and len(fired) > self.wta_k:
            fired.sort(key=lambda k: -v[k] if refr[k] == 0 else 0.0)
            dropped = fired[self.wta_k:]
            fired = fired[: self.wta_k]
            for k in dropped:
                v[k] = 0.0
                refr[k] = 1
        if force:
            have = set(fired)
            for k in force:
                k = int(k)
                if k not in have:
                    fired.append(k)
                    v[k] = 0.0
                    refr[k] = refr_steps
        self._last_spikes = fired
        rate = len(fired) / self.n
        error = self.target_hz - rate * 1000.0
        g = self.gain * _exp(self.eta_homeostasis * error)
        self.gain = 0.05 if g < 0.05 else (20.0 if g > 20.0 else g)
        d = self.drive + self.eta_homeostasis * 1.5 * error
        self.drive = -0.5 if d < -0.5 else (1.8 if d > 1.8 else d)
        return fired

    def rates(self) -> list[float]:
        out = [0.0] * self.n
        for k in self._last_spikes:
            out[k] = 1.0
        return out


def _exp(x: float) -> float:
    """``math.exp`` guarded to avoid overflow on long runs."""
    import math

    if x > 50.0:
        return 1.0e21
    if x < -50.0:
        return 0.0
    return math.exp(x)


def zeros(n: int):
    """A fresh zero-filled current buffer."""
    return [0.0] * n


def group_counts(spike_idx, n_groups: int, group_size: int):
    """Number of spikes per output group, as a flat list."""
    counts = [0.0] * n_groups
    for k in spike_idx:
        g = k // group_size
        if g < n_groups:
            counts[g] += 1.0
    return counts


def build_random(n_pre: int, n_post: int, fan_in: int, rng: "Rng",
                 scale: float = 1.0):
    """Random projection ``(n_pre -> n_post)`` as plain lists.

    Loops once per synapse, so keep this for small networks; the NumPy
    backend builds the same structure vectorised.
    """
    total = n_post * fan_in
    pre = [rng.randint(0, n_pre) for _ in range(total)]
    post = [dst for dst in range(n_post) for _ in range(fan_in)]
    weight = rng.normal_list(total, 0.0, scale)
    return pre, post, weight


def build_sensor(n_inputs: int, n_neurons: int, span: int, rng: "Rng",
                 scale: float = 1.0):
    """Sparse sensor projection with a bounded receptive field."""
    total = n_inputs * span
    pre = [i for i in range(n_inputs) for _ in range(span)]
    post = [rng.randint(0, n_neurons) for _ in range(total)]
    weight = rng.normal_list(total, 0.0, scale)
    return pre, post, weight


def add_block(dst, src) -> None:
    """``dst[:len(src)] += src`` — used when a projection targets one region."""
    for i, v in enumerate(src):
        dst[i] += v
