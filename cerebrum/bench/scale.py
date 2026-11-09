"""Scale benchmark: how big a brain this machine can actually run.

Measures step cost and resident memory for a sparse network at several sizes
and firing rates, using the same inner loop the engine uses: event-driven
propagation over the presynaptic neurons that fired, then a vectorised
membrane update over every neuron.

Run::

    python -m cerebrum.bench.scale
"""

from __future__ import annotations

import time

import numpy as np


def _rss_mb() -> float:
    """Resident set size in MB (Windows), -1 where unavailable."""
    try:
        import ctypes

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        k32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(PMC),
            ctypes.c_uint,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        info = PMC()
        info.cb = ctypes.sizeof(PMC)
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(info), info.cb)
        return info.WorkingSetSize / 1024**2
    except Exception:
        return -1.0


def measure(n_neurons: int, fan_in: int = 20, rate: float = 0.01,
            n_steps: int = 40, seed: int = 0) -> dict:
    """Build a sparse network of ``n_neurons`` and time ``n_steps`` steps."""
    rng = np.random.default_rng(seed)
    n_syn = n_neurons * fan_in

    pre = rng.integers(0, n_neurons, size=n_syn, dtype=np.int32)
    post = np.repeat(np.arange(n_neurons, dtype=np.int32), fan_in)
    weight = rng.normal(0.0, 0.045, size=n_syn).astype(np.float32)
    order = np.argsort(pre, kind="stable").astype(np.int64)
    indptr = np.zeros(n_neurons + 1, np.int64)
    np.cumsum(np.bincount(pre, minlength=n_neurons), out=indptr[1:])
    del pre

    v = np.zeros(n_neurons, np.float32)
    threshold = np.ones(n_neurons, np.float32)
    bias = rng.normal(0.13, 0.19, n_neurons).astype(np.float32)
    adapt = np.zeros(n_neurons, np.float32)
    refractory = np.zeros(n_neurons, np.int8)

    n_spike = max(1, int(n_neurons * rate))

    def step(spikes: np.ndarray) -> np.ndarray:
        current = np.zeros(n_neurons, np.float32)
        lo = indptr[spikes]
        counts = indptr[spikes + 1] - lo
        total = int(counts.sum())
        if total:
            offsets = np.repeat(
                lo - np.concatenate(([0], np.cumsum(counts)[:-1])), counts
            )
            sids = order[np.arange(total, dtype=np.int64) + offsets]
            np.add.at(current, post[sids], weight[sids])
        v[:] = v + (-v + current + bias - 0.3 * adapt) / 10.0
        v[refractory > 0] = 0.0
        np.maximum(refractory - 1, 0, out=refractory)
        fired = v >= threshold
        v[fired] = 0.0
        refractory[fired] = 3
        return np.nonzero(fired)[0]

    spikes = rng.choice(n_neurons, n_spike, replace=False)
    for _ in range(5):          # warm up caches
        step(spikes)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        step(spikes)
    ms = (time.perf_counter() - t0) / n_steps * 1000

    return {
        "neurons": n_neurons,
        "synapses": n_syn,
        "rate": rate,
        "ms_per_step": round(ms, 2),
        "steps_per_s": round(1000.0 / ms, 1),
        "rss_mb": round(_rss_mb()),
    }


def main() -> None:
    print("%9s %6s %5s %10s %10s %8s"
          % ("neurons", "synapses", "rate", "ms/step", "steps/s", "RSS MB"))
    for n in (10_000, 100_000, 1_000_000):
        for rate in (0.01, 0.03):
            r = measure(n, rate=rate)
            print("%9d %6d %4.0f%% %10.2f %10.0f %8d"
                  % (r["neurons"], r["synapses"], r["rate"] * 100,
                     r["ms_per_step"], r["steps_per_s"], r["rss_mb"]))


if __name__ == "__main__":
    main()
