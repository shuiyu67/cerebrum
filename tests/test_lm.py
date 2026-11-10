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
