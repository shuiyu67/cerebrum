"""Neuron sheets: the substrate every anatomical region is built on.

Why this exists
---------------
The engine's first generation gave every neuron an index and nothing else.
Index 0 had no relationship to index 1, so "neighbour" was undefined, so no
patch of cells could ever become a functional unit. Measured on an
800-neuron flat population with competitive input learning: adjacent neurons
agreed on their preferred stimulus 51% of the time against a 50% chance
baseline — i.e. preferences were scattered noise, not territory.

A cortical area is not a bag of cells. It is a *sheet*: cells sit at
positions, they wire mostly to nearby cells, they are inhibited mostly by
nearby interneurons, and that is exactly what lets neighbouring cells share
a fate and distant cells diverge. This class supplies those three things.

Everything here is a leaky integrate-and-fire sheet with:

* **positions** — a jittered 2-D grid, normalised to the unit square;
* **distance-decayed connectivity** — ``p = p_local * exp(-d^2 / 2 sigma^2)``,
  with four weight classes (E->E, E->I, I->E, I->I);
* **spatially interleaved inhibition** — inhibitory cells are scattered
  through the sheet rather than collected at one end, so inhibition has a
  locality;
* **per-neuron threshold homeostasis** — each cell regulates its own
  excitability, which is what keeps a self-organising map from leaving dead
  units behind;
* **afferent plasticity with a neighbourhood kernel** — the mechanism that
  turns spatial neighbours into functional neighbours.

Time step is 1 ms throughout.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def build_projection(n_post: int, n_pre: int, fan_in: int, weight: float,
                     rng, *, pos_pre: Optional[np.ndarray] = None,
                     pos_post: Optional[np.ndarray] = None,
                     topo: bool = False, sign: float = 1.0) -> np.ndarray:
    """Dense projection matrix ``(n_post, n_pre)``, applied as ``P @ spikes``.

    ``topo=True`` wires each target cell to its *nearest* source cells in
    sheet coordinates, which is how a channel-indexed nucleus (thalamus) maps
    onto a continuous sheet (cortex) without scrambling neighbourhoods. The
    default is uniform random fan-in, which is right for modulatory paths
    where no spatial relationship exists.
    """
    W = np.zeros((n_post, n_pre), np.float32)
    fan = max(1, min(int(fan_in), n_pre))
    if topo and pos_pre is not None and pos_post is not None:
        for j in range(n_post):
            d2 = ((pos_pre - pos_post[j]) ** 2).sum(axis=1)
            src = np.argpartition(d2, min(fan, n_pre - 1))[:fan]
            W[j, src] = weight
    else:
        for j in range(n_post):
            src = rng.choice(n_pre, fan, replace=False)
            W[j, src] = weight
    return W * np.float32(sign)


class NeuronSheet:
    """A 2-D sheet of leaky integrate-and-fire neurons.

    Parameters
    ----------
    name:
        Label used in reports and for cross-region addressing.
    n:
        Number of cells.
    shape:
        ``(rows, cols)`` for the position grid. ``None`` lays the cells out on
        a roughly square grid anyway; a 1-D sheet is expressible as
        ``(1, n)``.
    excitatory_frac:
        Fraction of cells that are excitatory. The rest are inhibitory
        interneurons, placed at random positions rather than blocked
        together, because clustered inhibition cannot carve local territory.
    p_local:
        Connection probability for two cells at the same position.
    sigma_conn:
        Spatial constant of the connectivity kernel, in sheet units.
    tau_m, v_threshold, refractory_steps:
        Membrane time constant (ms), firing threshold, and refractory period.
    g_adapt, tau_adapt:
        Spike-frequency adaptation strength and time constant.
    target_hz:
        Rate each cell's threshold controller aims for. Because a step is
        1 ms, this is also the per-step firing probability the controller
        drives towards.
    eta_th:
        Threshold homeostasis rate. Too high and the sheet chases noise; too
        low and dead cells stay dead. 1/2000 per step settles in a few
        thousand steps.
    w_scale:
        Overall synaptic gain.
    """

    __slots__ = (
        "name", "n", "shape", "pos", "is_inh", "n_exc", "n_inh",
        "W", "EE", "EI", "IE", "II",
        "v", "th", "bias", "adapt", "refractory", "rate_ema",
        "prev", "ext", "_zero",
        "tau_m", "g_adapt", "tau_adapt", "refractory_steps",
        "target_hz", "eta_th", "v_threshold", "w_scale", "w_norm",
        "th_max", "tau_rate", "noise_std", "_rng", "eta_down",
        "global_inh", "g", "g_min", "g_max",
        "spikes_total", "steps_total", "last_current",
    )

    def __init__(
        self,
        name: str,
        n: int,
        *,
        shape: Optional[tuple] = None,
        excitatory_frac: float = 0.8,
        p_local: float = 0.25,
        sigma_conn: float = 0.12,
        tau_m: float = 10.0,
        v_threshold: float = 1.0,
        refractory_steps: int = 2,
        g_adapt: float = 0.25,
        tau_adapt: float = 120.0,
        bias_mean: float = 0.10,
        bias_std: float = 0.08,
        target_hz: float = 6.0,
        eta_th: float = 1.5e-3,
        th_max: float = 6.0,
        noise_std: float = 0.045,
        eta_down: float = 1.2e-2,
        global_inh: float = 0.0,
        g_init: float = 1.0,
        g_min: float = 0.08,
        g_max: float = 5.0,
        tau_rate: float = 300.0,
        w_scale: float = 1.0,
        w_norm: float = 1.5,
        seed: int = 0,
    ) -> None:
        self.name = name
        self.n = int(n)
        if self.n < 1:
            raise ValueError("a sheet needs at least one cell")
        rng = np.random.default_rng(seed)

        self.shape = tuple(shape) if shape else self._auto_shape(self.n)
        self.pos = self._layout(self.shape, self.n, rng)

        n_inh = int(round(self.n * (1.0 - excitatory_frac)))
        n_inh = min(max(n_inh, 0), self.n)
        self.is_inh = np.zeros(self.n, bool)
        if n_inh:
            self.is_inh[rng.choice(self.n, n_inh, replace=False)] = True
        self.n_inh = int(self.is_inh.sum())
        self.n_exc = self.n - self.n_inh

        self.tau_m = float(tau_m)
        self.v_threshold = float(v_threshold)
        self.refractory_steps = int(refractory_steps)
        self.g_adapt = float(g_adapt)
        self.tau_adapt = float(tau_adapt)
        self.target_hz = float(target_hz)
        self.eta_th = float(eta_th)
        self.th_max = float(th_max)
        self.noise_std = float(noise_std)
        self.eta_down = float(eta_down)
        self.global_inh = float(global_inh)
        # Per-cell synaptic gain. Homeostasis is multiplicative on the input
        # rather than additive on the threshold: a gain controller reduces
        # error geometrically and cannot wind up, while an integrating
        # threshold controller overshoots its target, saturates at the cap,
        # and then has to unwind — measured, L2/3 oscillated between 5 and 60
        # spikes per step with a period near 600 ms, forever.
        self.g_min = float(g_min)
        self.g_max = float(g_max)
        self.g = np.full(self.n, float(g_init), np.float32)
        self._rng = rng
        self.tau_rate = float(tau_rate)
        self.w_scale = float(w_scale)
        self.w_norm = float(w_norm)

        self.v = np.zeros(self.n, np.float32)
        self.th = np.full(self.n, self.v_threshold, np.float32)
        self.bias = rng.normal(bias_mean, bias_std, self.n).astype(np.float32)
        self.adapt = np.zeros(self.n, np.float32)
        self.refractory = np.zeros(self.n, np.int8)
        self.rate_ema = np.zeros(self.n, np.float32)
        self.prev = np.empty(0, np.int64)
        self.ext = np.zeros(self.n, np.float32)
        self._zero = np.zeros(self.n, np.float32)
        self.spikes_total = 0
        self.steps_total = 0
        self.last_current = self._zero

        self._build_connectivity(p_local, sigma_conn, rng)

    # -- geometry --------------------------------------------------------

    @staticmethod
    def _auto_shape(n: int) -> tuple:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        return (rows, cols)

    @staticmethod
    def _layout(shape: tuple, n: int, rng) -> np.ndarray:
        """Cell positions on a jittered grid, normalised to the unit square.

        A perfect grid would make every cell's neighbourhood identical; a
        little jitter is what breaks the symmetry so the map can actually
        choose an orientation instead of sitting on a knife edge.
        """
        rows, cols = shape
        ys, xs = np.mgrid[0:rows, 0:cols]
        xs = xs.ravel().astype(np.float64)
        ys = ys.ravel().astype(np.float64)
        span_x = max(1.0, cols - 1.0)
        span_y = max(1.0, rows - 1.0)
        pts = np.stack([
            xs / span_x + rng.normal(0, 0.35 / max(1.0, span_x), xs.size),
            ys / span_y + rng.normal(0, 0.35 / max(1.0, span_y), ys.size),
        ], axis=1)
        return np.clip(pts, 0.0, 1.0)[:n]

    def distance_matrix(self) -> np.ndarray:
        """Euclidean distance between every pair of cells, in sheet units."""
        d = self.pos[:, None, :] - self.pos[None, :, :]
        return np.sqrt((d ** 2).sum(axis=2))

    # -- wiring ----------------------------------------------------------

    def _build_connectivity(self, p_local: float, sigma_conn: float, rng) -> None:
        """Distance-decayed, E/I-separated, sign-correct recurrent matrix.

        ``W[post, pre]`` so that a single matmul-free gather computes the
        synaptic current: ``W[:, spikes].sum(axis=1)``.
        """
        d = self.distance_matrix()
        p = p_local * np.exp(-(d ** 2) / (2.0 * sigma_conn ** 2))
        np.fill_diagonal(p, 0.0)
        mask = rng.random((self.n, self.n)) < p

        pre_inh = self.is_inh[None, :]          # broadcast over posts
        post_inh = self.is_inh[:, None]

        # Connection strength: excitatory cells deliver positive current,
        # inhibitory cells negative. Inhibitory synapses are made stronger
        # because there are 4x fewer of them and they have to hold the sheet
        # in a sparse regime on their own.
        # Sign, not just magnitude: an inhibitory cell must deliver
        # negative current. Writing 1.9 here made every interneuron in every
        # sheet excitatory, which is why the sheets were so hard to calm down.
        mag = np.where(pre_inh, -1.9, 0.55).astype(np.float32)
        mag = mag * np.float32(self.w_scale)

        # Scale down per cell so a cell with many neighbours does not drown
        # in summed current: total input is normalised by expected fan-in.
        fan_in = mask.sum(axis=1).astype(np.float32) + 1.0
        norm = np.float32(self.w_norm) / np.sqrt(fan_in)

        self.W = (mask.astype(np.float32) * mag * norm[:, None]).astype(np.float32)
        np.fill_diagonal(self.W, 0.0)

        self.EE = mask & ~pre_inh & ~post_inh
        self.EI = mask & pre_inh & ~post_inh
        self.IE = mask & ~pre_inh & post_inh
        self.II = mask & pre_inh & post_inh

    def connect_from(self, other: "NeuronSheet", fan_in: int, weight: float,
                     *, topo: bool = False, jitter: float = 1.0,
                     rng: Optional[np.random.Generator] = None,
                     sign: float = 1.0) -> np.ndarray:
        """Build a projection ``other -> self``.

        ``topo`` maps by relative position instead of random sampling: a cell
        in the source at ``(x, y)`` drives cells near ``(x, y)`` in the
        target. Topographic projections are what keep a map a *map* across
        areas; random ones are faster to build and fine for modulatory
        paths. Returns the sparse weight matrix ``(self.n, other.n)``.
        """
        rng = rng if rng is not None else np.random.default_rng(12345)
        W = np.zeros((self.n, other.n), np.float32)
        if topo:
            # nearest-source assignment, then a local spread
            for j in range(self.n):
                p = self.pos[j]
                d2 = ((other.pos - p) ** 2).sum(axis=1)
                centre = int(np.argmin(d2))
                pick = rng.integers(0, other.n, fan_in) // 1
                near = np.argsort(d2)[: max(1, fan_in // 2)]
                src = np.concatenate([[centre], near, pick])[:fan_in]
                src = np.unique(src)
                W[j, src] = weight
        else:
            for j in range(self.n):
                src = rng.choice(other.n, min(fan_in, other.n), replace=False)
                W[j, src] = weight
        return W * np.float32(sign)

    # -- dynamics --------------------------------------------------------

    def step(self, ext: Optional[np.ndarray] = None, *,
             force: Optional[np.ndarray] = None,
             syn_extra: Optional[np.ndarray] = None) -> np.ndarray:
        """Advance 1 ms. Returns the indices that fired.

        ``ext`` is external current (already scaled to threshold units: 1.0
        is roughly enough to fire an isolated cell in one step). ``force``
        makes a set of cells fire regardless, which is how a region replays a
        pattern it is not currently being shown.
        """
        if ext is None:
            ext_arr = self._zero
        else:
            ext_arr = np.asarray(ext, np.float32)
            if ext_arr.shape[0] != self.n:
                raise ValueError(
                    f"{self.name}: expected {self.n} inputs, got {ext_arr.shape[0]}"
                )

        prev = self.prev
        if prev.size:
            syn = self.W[:, prev].sum(axis=1)
        else:
            syn = self._zero.copy()
        if self.global_inh > 0.0:
            # Pooled feedback inhibition, the basket-cell pool. Local
            # inhibition is too slow to stop a population burst: by the time
            # the surround reacts, the burst has already recruited the sheet.
            # A graded pool reads the *total* activity and pushes back in
            # proportion, which is what keeps a recurrent sheet in the
            # balanced regime instead of flipping between silence and
            # synchronous discharge (measured: L5 alternated between 0 and
            # 170 cells per step before this term).
            syn = syn - np.float32(self.global_inh * prev.size / self.n)
        if syn_extra is not None:
            syn = syn + syn_extra

        v = self.v
        drive = (syn + ext_arr) * self.g + self.bias
        v += (-v + drive - self.g_adapt * self.adapt) / self.tau_m
        if self.noise_std > 0.0:
            # Membrane noise. This is not decoration: without it the LIF
            # threshold is a step function, so a cell's rate jumps from 0 to
            # 80 Hz as its input crosses threshold and no threshold
            # controller can hold it at a target — measured, every sheet
            # eventually pinned its threshold at the cap and went permanently
            # silent. Noise turns the step into a sigmoid, which is what real
            # neurons have and what makes rate control possible.
            v += self.noise_std * self._rng.standard_normal(self.n).astype(np.float32)
        v[self.refractory > 0] = 0.0
        np.maximum(self.refractory - 1, 0, out=self.refractory)

        fired = v >= self.th
        if force is not None and len(force):
            fired[np.asarray(force, np.int64)] = True

        v[fired] = 0.0
        self.refractory[fired] = self.refractory_steps
        self.adapt += (fired.astype(np.float32) - self.adapt) * (1.0 / self.tau_adapt)

        # Per-neuron threshold homeostasis. A single sheet-wide gain cannot
        # keep a map alive: cells that win the competition drive their
        # neighbours silent, and those neighbours never recover because they
        # never fire. Letting each cell move its own threshold means a
        # silenced cell lowers its threshold until it can fire again.
        self.rate_ema += (fired.astype(np.float32) - self.rate_ema) * (1.0 / self.tau_rate)
        target = self.target_hz / 1000.0
        err = np.clip(self.rate_ema / target - 1.0, -3.0, 3.0)
        # Multiplicative update. Silent cells get their gain raised, loud ones
        # get it lowered, and the correction is proportional to how far off
        # they are rather than accumulating without bound.
        self.g *= np.exp(-self.eta_th * err)
        np.clip(self.g, self.g_min, self.g_max, out=self.g)

        idx = np.nonzero(fired)[0]
        self.prev = idx
        self.last_current = syn
        self.spikes_total += int(idx.size)
        self.steps_total += 1
        return idx

    def rate_hz(self, window: int = 0) -> float:
        """Mean firing rate in Hz (10 spikes/s at 1 ms steps = 6 Hz)."""
        if self.steps_total == 0:
            return 0.0
        return self.spikes_total / self.steps_total / self.n * 1000.0

    def activity(self, steps: int = 1, ext: Optional[np.ndarray] = None,
                 mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Run ``steps`` and return per-cell spike counts (for readout)."""
        acc = np.zeros(self.n, np.float32)
        for _ in range(steps):
            idx = self.step(ext)
            if idx.size:
                acc[idx] += 1.0
        if mask is not None:
            acc = acc * mask
        return acc / max(1, steps)

    def reset(self) -> None:
        """Clear transient electrical state, keeping thresholds and weights."""
        self.v.fill(0.0)
        self.adapt.fill(0.0)
        self.refractory.fill(0)
        self.prev = np.empty(0, np.int64)

    # -- plasticity ------------------------------------------------------

    def learn_map(self, afferent: np.ndarray, x: np.ndarray, *, lr: float = 0.05,
                  sigma: Optional[float] = None, eps: float = 1e-6) -> float:
        """One step of neighbourhood-competitive afferent learning (a SOM).

        ``afferent`` is ``(n_cells, n_inputs)``; ``x`` is the input vector.

        The winner is the cell whose afferent weights best match ``x``. The
        winner *and its spatial neighbours* move towards ``x``, weighted by a
        Gaussian of distance. Updating only the winner gives a scattered,
        non-topographic arrangement; updating the neighbourhood is the whole
        reason neighbouring cells end up preferring neighbouring inputs, i.e.
        the reason a functional area can appear without anyone drawing its
        border.

        Returns the winning cell's match score (``0..1``).
        """
        x = np.asarray(x, np.float32)
        norm = float(np.linalg.norm(x)) + eps
        xn = x / norm
        score = afferent @ xn
        winner = int(np.argmax(score))

        if sigma is None:
            sigma = 0.15
        if sigma <= 1e-6:
            # Winner-take-all: only the matching cell learns. Kept as an
            # explicit mode because it is the control condition for the map
            # experiments — it is what a competitive rule does *without* a
            # neighbourhood, and it produces a scattered, non-topographic
            # arrangement (measured adjacency agreement 0.50, i.e. chance).
            h = np.zeros(self.n, np.float32)
            h[winner] = 1.0
        else:
            d2 = ((self.pos - self.pos[winner]) ** 2).sum(axis=1)
            h = np.exp(-d2 / (2.0 * sigma ** 2)).astype(np.float32)
        afferent += (lr * h)[:, None] * (xn[None, :] - afferent)
        # renormalise so the dot product stays a bounded similarity
        afferent /= (np.linalg.norm(afferent, axis=1, keepdims=True) + eps)
        return float(score[winner])

    def learn_hebbian(self, pre_idx: np.ndarray, lr: float = 0.01,
                      dopamine: float = 1.0, cutoff: float = 0.02) -> int:
        """Three-factor Hebbian update on the sheet's own recurrent matrix.

        Potentiation follows ``pre * post`` and is gated by a third factor
        (dopamine or a teaching signal). Without the gate, every co-active
        pair strengthens and the matrix saturates into a single global
        attractor; the gate is what makes reinforcement selective.
        """
        if pre_idx.size == 0 or self.prev.size == 0:
            return 0
        post = self.prev
        rows = self.W[post][:, pre_idx]
        delta = lr * np.float32(dopamine) * rows
        self.W[np.ix_(post, pre_idx)] += delta
        np.clip(self.W, -4.0, 4.0, out=self.W)
        # Keep Dales law: an inhibitory cell only ever inhibits. Without this
        # the Hebbian term flips a few dozen synapses to positive and the
        # sheet loses its brake (measured: rate ran away to 40 Hz).
        if self.n_inh:
            # Fancy indexing produces a copy, so ``out=`` would write into a
            # temporary and silently do nothing. Assign the result back.
            self.W[:, self.is_inh] = np.minimum(self.W[:, self.is_inh], 0.0)
        return int((np.abs(delta) > cutoff).sum())

    def summary(self) -> str:
        kind = f"{self.n_exc}E/{self.n_inh}I"
        return (f"Sheet({self.name}, n={self.n}, {kind}, "
                f"shape={self.shape}, rate={self.rate_hz():.2f} Hz)")
