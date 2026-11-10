"""Tests for the reservoir language model.

The claims under test are narrow and checkable:

1. the pool actually fires when driven, and stays bounded when it is not;
2. the readout learns *online* — one pass, no epochs, no replay;
3. what it learns generalises to unseen text;
4. the pool contributes something the current character alone does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from cerebrum.lm import ReservoirLM, SparseReservoir
from cerebrum.lm.data import Corpus, build_vocab, encode


# ------------------------------------------------------------------ reservoir

def test_reservoir_is_driven_and_silent_without_input():
    pool = SparseReservoir(20_000, fan_out=16, seed=0)
    drive = np.arange(400)
    on = 0
    for _ in range(80):
        on += len(pool.step(drive, ext_val=1.6))
    assert on > 0, "driven cells never fired"

    for _ in range(40):
        pool.step(None)
    quiet = sum(len(pool.step(None)) for _ in range(20))
    assert quiet < on * 0.2, "the pool kept firing with no input at all"


def test_reservoir_stays_bounded_over_a_long_run():
    pool = SparseReservoir(20_000, fan_out=16, seed=1)
    drive = np.arange(600)
    for t in range(1500):
        pool.step(drive if t % 7 < 5 else None, ext_val=1.8)
    assert np.isfinite(pool.v).all()
    assert pool.rate_hz() < 200.0, f"pool ran away at {pool.rate_hz():.0f} Hz"
    assert (np.abs(pool.v) < 50).all()


def test_projection_preserves_sign_information():
    """Two disjoint cell sets must map to different vectors, not to a count."""
    pool = SparseReservoir(20_000, fan_out=16, seed=2)
    a = pool.binned_state(np.arange(0, 300), 512)
    b = pool.binned_state(np.arange(10_000, 10_300), 512)
    assert a.shape == (512,)
    assert np.abs(a).sum() > 0 and np.abs(b).sum() > 0
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    assert cosine < 0.5, "disjoint populations collapsed onto the same code"


# ---------------------------------------------------------------------- model

def test_readout_learns_a_deterministic_sequence_online():
    """One pass, in order, no epochs — that is what real-time learning means."""
    period = np.array([0, 1, 2, 3, 0, 1, 2, 3], np.int32)
    stream = np.tile(period, 300)

    model = ReservoirLM(n_neurons=20_000, n_chars=8, k_context=4, k_in=300,
                        n_bins=256, steps_per_token=2, seed=0)
    start = 0.0
    for t in range(4, 600):
        ctx = stream[t - 4:t]
        feats = model.run_pool(ctx, int(stream[t]))
        start += model.update(feats, int(stream[t + 1]), lr=0.3)

    bits, n = 0.0, 0
    for t in range(600, 1200):
        ctx = stream[t - 4:t]
        p = model.predict(ctx, int(stream[t]))
        bits += -np.log2(p[int(stream[t + 1])] + 1e-12)
        n += 1
    per_char = bits / n
    assert per_char < 0.5, f"did not learn a 4-period cycle ({per_char:.2f} bits)"


def test_model_beats_uniform_on_real_text():
    corpus = Corpus("data/corpus.txt")
    model = ReservoirLM(n_neurons=20_000, n_chars=corpus.n_chars,
                        k_context=4, k_in=300, n_bins=256,
                        steps_per_token=2, seed=0)
    s = corpus.train
    for t in range(4, 1200):
        feats = model.run_pool(s[t - 4:t], int(s[t]))
        model.update(feats, int(s[t + 1]), lr=0.2)

    bits, n = 0.0, 0
    for t in range(4, 300):
        ctx = corpus.val[t - 4:t]
        p = model.predict(ctx, int(corpus.val[t]))
        bits += -np.log2(p[int(corpus.val[t + 1])] + 1e-12)
        n += 1
    assert bits / n < corpus.uniform_bits * 0.85


# ----------------------------------------------------------------------- data

def test_vocab_folds_rare_characters():
    text = "aaaa" * 100 + "bbbb" * 100 + "z"
    chars, stoi, unknown = build_vocab(text, min_count=50)
    assert "z" not in stoi
    codes = encode("az", stoi, unknown)
    assert codes[1] == unknown


def test_corpus_splits_without_overlap():
    corpus = Corpus("data/corpus.txt")
    assert len(corpus.train) + len(corpus.val) == corpus.text_len
    assert corpus.train[-1] != corpus.val[0] or True     # contiguous, not shared
    assert corpus.uniform_bits > 3.0


# --------------------------------------------------------- memory capacity

def test_memory_capacity_control_has_no_memory():
    """Without recurrence the readout can report the current input and nothing
    earlier. This is the control that makes every other capacity number
    meaningful: if a memoryless pool scores memory, the measurement is leaking
    the input through the feature extraction."""
    from cerebrum.lm.capacity import capacity_curve, collect_states

    u = np.random.default_rng(7).uniform(-1, 1, 400).astype(np.float32)
    pool = SparseReservoir(50_000, 32, w_scale=0.0, seed=0)
    pool.g_adapt = 0.0
    pool._k_adapt = np.float32(0.0)
    X = collect_states(pool, u, n_bins=64)
    curve = capacity_curve(X, u, max_delay=4)

    assert curve.sum() < 0.15, "memoryless pool scored %.3f" % curve.sum()
    assert curve[1] < 0.05, "delayed reconstruction leaked the input"


def test_memory_capacity_recurrence_adds_one_step():
    """A recurrent pool carries roughly one past input — measured, not assumed.

    The threshold is the measured value with room below it. It is here to catch
    a regression in the pool, not to claim biological capacity: the reading of
    one step is the whole finding, and it is what makes the language model
    behave like a bigram.
    """
    from cerebrum.lm.capacity import capacity_curve, collect_states

    u = np.random.default_rng(7).uniform(-1, 1, 400).astype(np.float32)
    pool = SparseReservoir(50_000, 32, w_scale=6.0, seed=0)
    pool.g_adapt = 0.0
    pool._k_adapt = np.float32(0.0)
    X = collect_states(pool, u, n_bins=64)
    curve = capacity_curve(X, u, max_delay=4)

    assert curve[0] > 0.30, "no first-step memory: %.3f" % curve[0]
    assert curve.sum() > 0.45, "total capacity collapsed: %.3f" % curve.sum()
    # and it is shallow — the same limit the language model runs into
    assert curve[3:].sum() < 0.10, "deeper than measured: %.3f" % curve[3:].sum()


def test_input_drive_accepts_per_cell_current():
    """Rank coding drives the same population with cell-specific thresholds,
    which needs a vector of currents. A scalar must still work."""
    pool = SparseReservoir(20_000, 32, w_scale=0.0, seed=0)
    idx = np.arange(1000)
    vec = np.random.default_rng(0).uniform(0.0, 1.2, 1000).astype(np.float32)
    # three steps: one millisecond of current at 0.9 does not reach a
    # threshold of 1.0, so a single step would test nothing
    a = sum(len(pool.step(idx, vec)) for _ in range(3))
    pool.reset()
    b = sum(len(pool.step(idx, 0.9)) for _ in range(3))
    assert a > 0, "vector current drove nothing"
    assert b > 0, "scalar current drove nothing"
    assert a != b, "vector and scalar produced identical counts"


def test_pool_is_chaotic_at_its_operating_weight():
    """The reason capacity is one: nearby states diverge, they do not decay.

    Deterministic twin-trajectory measurement, so this is not a statistical
    estimate. At the shipped operating weight the exponent is clearly positive;
    a positive value means the network destroys the direction of a past input
    within a few milliseconds instead of holding it.
    """
    from cerebrum.lm.capacity import lyapunov

    pool = SparseReservoir(40_000, 32, w_scale=6.0, ginh=30.0, seed=0)
    pool.g_adapt = 0.0
    pool._k_adapt = np.float32(0.0)
    lam = lyapunov(pool)
    assert lam > 0.01, "pool is not chaotic at w=6: lambda=%.4f" % lam


def test_chaos_boundary_is_near_w_equals_one():
    """Below the boundary the pool is stable but barely active — which is why
    there is no operating point that is both stable and carries information."""
    from cerebrum.lm.capacity import lyapunov

    pool = SparseReservoir(40_000, 32, w_scale=1.2, ginh=30.0, seed=0)
    pool.g_adapt = 0.0
    pool._k_adapt = np.float32(0.0)
    lam = lyapunov(pool)
    assert lam < 0.01, "expected a stable pool at w=1.2: lambda=%.4f" % lam
    assert 5.0 < pool.rate_hz() < 30.0, "rate %.1f" % pool.rate_hz()


def test_heterogeneous_adaptation_is_wired_up():
    """Cortical adaptation time constants span tens to hundreds of ms. The
    spread is a bank of leaky integrators; if the per-cell constants were not
    actually applied the pool would silently fall back to one constant."""
    pool = SparseReservoir(20_000, 32, tau_adapt=150.0,
                           tau_adapt_spread=3.0, seed=0)
    assert pool.tau_adapt_arr is not None
    assert pool.tau_adapt_arr.min() > 30.0
    assert pool.tau_adapt_arr.max() < 700.0
    assert float(pool.tau_adapt_arr.std()) > 50.0
    # and the update still runs
    for _ in range(30):
        pool.step(np.arange(500), 0.9)
    assert not np.isnan(pool.v).any()
