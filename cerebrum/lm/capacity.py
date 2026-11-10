"""Memory capacity of the spiking pool — how many past inputs does it hold?

Jaeger's memory capacity (2001) is the standard measure of what a reservoir can
store, and it answers a question a language-model loss cannot: *how far back
does the state carry information*, independent of any task or vocabulary.

The definition
--------------
For a random input stream ``u(t)`` and a linear readout trained to reconstruct
the delayed input ``u(t-k)``::

    MC_k = corr^2( u(t-k), u_hat_k(t) )
    MC   = sum_k MC_k

Each ``MC_k`` is at most 1, so ``MC`` is bounded by the readout dimension. A
pure delay line of length D scores exactly D. A memoryless system scores about
1: it can report the current input and nothing earlier.

Why this instead of another LM run
----------------------------------
The LM loss conflates capacity, readout quality and the difficulty of the
particular corpus. MC isolates capacity. If the curve collapses at k=2, no
readout and no corpus will make the LM competitive and the fix has to be
structural. If it reaches k=15, the LM's problem is somewhere else.

Three things that had to be fixed before any of this measured anything
----------------------------------------------------------------------
1. **Input coding.** Uniform current onto a driven population locks every cell
   into the same refractory phase, so the spike count reflects the phase and
   not the input (measured: correlation between drive magnitude and spike
   count 0.48, no readout dimension above 0.31, k=0 reconstruction 0.03).
   Replaced with rank coding: each input cell has its own fixed threshold and
   is driven when ``u`` exceeds it. Nested populations, monotone in ``u``.

2. **Input coverage.** 2% of the pool driven left ~54 cells firing per step at
   100k neurons, which is 0.05% activity — the projection was pure noise.
   Coverage is now 10% of the pool.

3. **Hold time.** A new input every millisecond against a 20 ms membrane and a
   2 ms refractory means the state samples an aliased signal. Measured, one
   step per input gave k=0 reconstruction 0.04; eight steps per input gives
   0.85 for the *memoryless* control. Inputs are held for ``HOLD`` steps, and
   the delay axis is counted in held inputs — the same convention the language
   model uses with ``steps_per_token``.

The result
----------
Measured on the pool used by the language model:

* Control (no recurrence): MC 0.06, depth 0 — the measurement is clean.
* Same 32M synapse budget, reshaped: 1Mx32 -> 0.53 (depth 2), 500kx64 -> 0.44,
  250kx128 -> 0.50, 125kx256 -> 0.45. **More synapses per cell does not buy
  depth.** The weights were normalised by 1/sqrt(fan_out), which holds the
  spectral radius fixed, so the dynamics did not change.
* Sweeping the spectral radius at 100k: peaks at w = 6-8 with MC 0.64-0.67 and
  MC_2 never above 0.09. Beyond that the pool saturates and capacity falls.
* Removing spike-frequency adaptation doubles MC_1 (0.21 -> 0.39 at w=4). The
  adaptation current is actively erasing the state.
* Membrane time constant 20 / 40 / 80 ms: depth stays at 1-2 and MC_1 falls as
  tau grows, because a slower membrane tracks the input less well.
* Balanced regime (small weights, strong inhibition, 0.5% target activity):
  capacity collapses to 0.05-0.12. A quiet pool has no memory either.

**The pool holds one input.** Every parameter that could plausibly extend it
has been swept, and the depth does not move. This is not a tuning failure; it
is the operating point of a random recurrent network with single-spike-triggered
firing, where a cell needs only two or three coincident inputs to cross
threshold, so it never integrates.

It also closes the loop on the language model: a pool with one input of memory
*is* a bigram with extra noise, which is exactly what the LM measured — 4.68
bits against 4.36 for a counting table, and delayed recall that dies at k=3.

The control arm
---------------
``w_scale=0`` removes recurrence. That pool must score ~1 (current input only,
zero for every ``k >= 1``). If it scores memory, information is leaking through
the readout and the numbers mean nothing.
"""

from __future__ import annotations

import numpy as np

from .reservoir import SparseReservoir

HOLD = 6          # milliseconds each input value is applied
COVER = 0.10      # fraction of the pool wired to the input
N_BINS = 256      # readout dimension
EMA = 0.2         # readout integration, ~5 ms


def _input_populations(n_in: int, seed: int = 0):
    """Two disjoint rank-coded populations plus their thresholds."""
    pos = np.arange(0, n_in)
    neg = np.arange(n_in, 2 * n_in)
    thr = np.random.default_rng(seed).uniform(0.0, 1.0, n_in)
    return pos, neg, thr


def collect_states(pool: SparseReservoir, u: np.ndarray, *, hold: int = HOLD,
                   cover: float = COVER, n_bins: int = N_BINS,
                   ema: float = EMA, drive: float = 0.9,
                   seed: int = 0) -> np.ndarray:
    """Run ``u`` through the pool, one state vector per input value.

    The state is an exponential moving average of the signed projection: a
    downstream cell integrates over a synaptic time constant rather than
    reading a single millisecond of spikes.
    """
    n_in = max(1, int(pool.n * cover / 2))
    pos, neg, thr = _input_populations(n_in, seed)

    X = np.zeros((u.size, n_bins), np.float32)
    state = np.zeros(n_bins, np.float32)
    for t in range(u.size):
        val = float(u[t])
        ext = pos[thr < val] if val > 0 else neg[thr < -val]
        for _ in range(hold):
            idx = pool.step(ext, ext_val=drive)
            if idx.size:
                state += (pool.binned_state(idx, n_bins) - state) * ema
            else:
                state *= (1.0 - ema)
        X[t] = state
    return X


def _ridge_r2(X: np.ndarray, y: np.ndarray, lam: float = 1e-2) -> float:
    """Squared correlation between ``y`` and its ridge reconstruction."""
    cut = int(X.shape[0] * 0.7)
    Xtr, ytr = X[:cut], y[:cut]
    Xte, yte = X[cut:], y[cut:]

    A = Xtr.T @ Xtr + lam * np.eye(X.shape[1], dtype=np.float32)
    w = np.linalg.solve(A, Xtr.T @ ytr)

    pred = Xte @ w
    if pred.std() < 1e-9 or yte.std() < 1e-9:
        return 0.0
    c = float(np.corrcoef(pred, yte)[0, 1])
    return max(0.0, c * c)


def capacity_curve(X: np.ndarray, u: np.ndarray,
                   max_delay: int = 20) -> np.ndarray:
    """``MC_k`` for k = 1..max_delay, in units of held inputs."""
    return np.array([_ridge_r2(X[k:], u[:-k])
                     for k in range(1, max_delay + 1)])


def calibrate_w_scale(fan_out: int, target_rate: float = 12.0,
                      n: int = 30_000, seed: int = 0) -> float:
    """Find the synaptic scale that puts this fan-out near the target rate.

    Total input current grows with ``fan_out`` at fixed activity, so holding
    ``w_scale`` fixed across fan-outs makes the wide configuration epileptic
    and the narrow one silent. ``1/sqrt(fan_out)`` is the right first guess,
    but the inhibitory pool and adaptation do not scale with it, so it is
    searched. Calibrated at a smaller pool: the number of active inputs a cell
    sees is ``fan_out x activity``, which does not depend on ``n``.
    """
    base = 8.0 * np.sqrt(32.0 / fan_out)
    u = np.random.default_rng(seed).uniform(-1, 1, 80).astype(np.float32)
    best, best_err = base, np.inf
    for mult in (0.5, 0.8, 1.2):
        ws = base * mult
        pool = SparseReservoir(n, fan_out, w_scale=ws, seed=seed)
        collect_states(pool, u, n_bins=32)
        err = abs(pool.rate_hz() - target_rate)
        if err < best_err:
            best, best_err = ws, err
        del pool
    return float(best)


def run(n: int, fan_out: int, *, seed: int = 0, T: int = 1000,
        max_delay: int = 20, w_scale: float | None = None,
        label: str = "") -> dict:
    """Measure the capacity curve for one (n, fan_out) configuration."""
    if w_scale is None:
        w_scale = calibrate_w_scale(fan_out, seed=seed)

    u = np.random.default_rng(seed + 7).uniform(-1.0, 1.0, T).astype(np.float32)
    pool = SparseReservoir(n, fan_out, w_scale=w_scale, seed=seed)
    X = collect_states(pool, u, seed=seed)
    rate = pool.rate_hz()
    mem = pool.memory_mb()
    del pool

    curve = capacity_curve(X, u, max_delay=max_delay)
    return {
        "label": label or "n=%d fan=%d" % (n, fan_out),
        "n": n, "fan_out": fan_out, "w_scale": float(w_scale),
        "synapses": n * fan_out, "memory_mb": mem, "rate_hz": rate,
        "curve": curve, "mc": float(curve.sum()),
        "depth": int((curve > 0.1).sum()),
    }


def report(results: list[dict], widths: int = 8) -> None:
    print("%-22s %-7s %-6s %-5s %-8s %-4s %s"
          % ("配置", "突触", "内存MB", "Hz", "总容量", "深度", "MC_1..MC_8"))
    for r in results:
        c = r["curve"]
        head = " ".join("%.2f" % v for v in c[:widths])
        print("%-22s %-7s %-6.0f %-5.1f %-8.2f %-4d %s"
              % (r["label"], "%.0fM" % (r["synapses"] / 1e6),
                 r["memory_mb"], r["rate_hz"], r["mc"], r["depth"], head))
    print()
    print("深度 = MC_k > 0.1 的最远 k；k 以「保持住的输入」为单位（每个保持 %d ms）" % HOLD)


def sweep(n: int = 100_000, T: int = 500, n_bins: int = 128,
          max_delay: int = 12) -> None:
    """Sweep every parameter that could plausibly buy memory depth."""
    u = np.random.default_rng(7).uniform(-1, 1, T).astype(np.float32)

    print("谱半径与适应（n=%d, fan=32）" % n)
    print("w_scale g_adapt | Hz    | MC总 深度 | MC_1 MC_2 MC_3")
    for ws in (2.0, 4.0, 6.0, 8.0, 12.0, 16.0):
        for ga in (0.30, 0.0):
            pool = SparseReservoir(n, 32, w_scale=ws, seed=0)
            pool.g_adapt = ga
            pool._k_adapt = np.float32(ga * pool.beta)
            X = collect_states(pool, u, n_bins=n_bins)
            c = capacity_curve(X, u, max_delay=max_delay)
            print("%7.1f %7.2f | %5.1f | %4.2f %4d | %s"
                  % (ws, ga, pool.rate_hz(), c.sum(), int((c > 0.1).sum()),
                     " ".join("%.2f" % v for v in c[:3])))


def main() -> None:
    TOTAL = 32_000_000
    print("=" * 82)
    print("Memory capacity — 同总突触预算（3200 万），改形状")
    print("=" * 82)
    print()

    results = [run(1_000_000, 32, w_scale=0.0, T=600, label="对照 零递归")]

    for fan in (32, 64, 128, 256):
        n = TOTAL // fan
        r = run(n, fan, T=1000, label="%d万神经元 x %d" % (n // 10000, fan))
        results.append(r)
        print("  完成 %s (w_scale=%.2f)" % (r["label"], r["w_scale"]))

    print()
    report(results)


if __name__ == "__main__":
    main()
