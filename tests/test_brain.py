"""Tests for the assembled brain.

These are behavioural assertions, not smoke tests. Each one states a claim the
architecture makes and fails if the claim stops being true. The thresholds are
the measured values with a margin, so a regression in any subsystem shows up
as a failure rather than as a slightly worse number nobody notices.
"""

from __future__ import annotations

import numpy as np
import pytest

from cerebrum.brain import (
    Amygdala, BasalGanglia, Body, Cerebellum, Cortex, Hippocampus,
    Hypothalamus, Neuromodulators, NeuronSheet, Prefrontal, Thalamus,
    WholeBrain, build_projection,
)
from cerebrum.brain.protocols import (
    protocol_one_trial_fear, protocol_pattern_completion,
    protocol_self_organisation, protocol_thalamic_gating,
)


# ------------------------------------------------------------------ substrate

def test_sheet_layout_is_exact():
    for n in (1, 7, 60, 120, 900):
        s = NeuronSheet("t", n, seed=0)
        assert s.pos.shape == (n, 2)
        assert s.W.shape == (n, n)
        assert s.n_exc + s.n_inh == n


def test_sheet_is_silent_without_input_but_responsive_with_it():
    s = NeuronSheet("t", 400, seed=1, target_hz=8.0)
    for _ in range(400):
        s.step(None)
    quiet = s.rate_hz()
    drive = np.zeros(400, np.float32)
    drive[:200] = 3.0
    for _ in range(400):
        s.step(drive)
    assert s.rate_hz() > quiet + 3.0, "sheet did not respond to drive"


def test_sheet_survives_long_run_and_stays_bounded():
    s = NeuronSheet("t", 300, seed=2)
    x = np.zeros(300, np.float32)
    x[:100] = 2.5
    for _ in range(3000):
        s.step(x)
    assert np.isfinite(s.v).all() and np.isfinite(s.W).all()
    assert s.rate_hz() < 120.0, "sheet ran away"
    assert s.g.min() >= s.g_min - 1e-6


def test_dales_law_holds_for_recurrent_weights():
    s = NeuronSheet("t", 300, seed=3)
    s.learn_hebbian(np.arange(20), lr=0.05, dopamine=2.0)
    if s.n_inh:
        assert (s.W[:, s.is_inh] <= 1e-6).all(), "inhibitory cells became excitatory"


# --------------------------------------------------------------- self-mapping

def test_cortex_partitions_itself_only_with_a_neighbourhood():
    """The central claim: functional areas appear without being specified."""
    out = protocol_self_organisation(epochs=25)
    control = out["control_no_neighbourhood"]["adjacency"]
    trained = out["with_neighbourhood"]["adjacency"]
    assert abs(control - 0.5) < 0.06, f"control should be chance, got {control:.3f}"
    assert trained > 0.85, f"neighbourhood learning should organise, got {trained:.3f}"
    assert out["with_neighbourhood"]["n_blocks"] < out["control_no_neighbourhood"]["n_blocks"] / 5


# ------------------------------------------------------------------ thalamus

def test_thalamus_relays_and_the_reticular_nucleus_gates():
    out = protocol_thalamic_gating()
    base = out["baseline_relay"]
    assert base["target"] > 0.01 and base["distractor"] > 0.01
    suppressed = out["feedback_on_distractor"]["distractor_kept_pct"]
    spared = out["feedback_on_distractor"]["target_kept_pct"]
    assert suppressed < 80.0, f"distractor not suppressed ({suppressed:.0f}%)"
    assert spared > 90.0, f"target should be spared ({spared:.0f}%)"
    assert out["reticular_lesioned"]["distractor_kept_pct"] > 95.0


def test_thalamus_rejects_wrong_width():
    with pytest.raises(ValueError):
        Thalamus(8, 4, seed=0).step(np.zeros(5, np.float32))


# ------------------------------------------------------------ basal ganglia

def test_basal_ganglia_selects_one_channel():
    bg = BasalGanglia(6, seed=0)
    c = np.full(6, 0.18)
    c[2] = 0.65
    for _ in range(4):
        gpi = bg.step(c)
    assert gpi[2] < 0.2, "the strongest proposal should be released"
    assert (gpi[[0, 1, 3, 4, 5]] > 0.4).all(), "the others should stay shut"
    assert bg.chosen == 2


def test_dopamine_biases_selection():
    def release(dopamine):
        bg = BasalGanglia(4, seed=0)
        c = np.full(4, 0.3)
        for _ in range(6):
            gpi = bg.step(c, dopamine=dopamine)
        return float((1.0 - gpi).mean())

    assert release(0.6) > release(0.0) > release(-0.6)


# --------------------------------------------------------------- hippocampus

def test_hippocampus_completes_a_partial_cue():
    out = protocol_pattern_completion()
    assert out["with_storage"]["completion"] > 0.4
    assert out["with_storage"]["completion"] > out["without_storage"]["completion"] + 0.2


def test_hippocampus_learns_in_one_exposure():
    rng = np.random.default_rng(0)
    h = Hippocampus(n_cortex=200, n_ec=200, n_dg=1000, n_ca3=600, seed=0)
    p = (rng.random(200) > 0.5).astype(np.float32)
    h.experience(p)
    assert len(h.codes) == 1
    rec = h.recall(p)
    assert rec["novelty"] < 0.2, "a stored pattern should not look novel"


# ------------------------------------------------------------------ amygdala

def test_one_pairing_is_enough_and_arousal_is_required():
    out = protocol_one_trial_fear()
    normal = out["normal_arousal"]
    assert normal["to_cs"] > 0.5
    assert normal["specificity"] > 0.5, "fear generalised to the unpaired cue"
    blocked = out["arousal_blocked"]
    assert blocked["to_cs"] < 0.05, "learning happened without arousal"


# ---------------------------------------------------------------- cerebellum

def test_cerebellum_ltd_reduces_the_trained_response():
    cue = np.zeros(64, np.float32)
    cue[:16] = 1.0

    def run(ltd, trials=25):
        cb = Cerebellum(64, n_granule=300, n_purkinje=40, n_deep=40,
                        ltd=ltd, seed=0)
        for _ in range(trials):
            cb.reset()
            for k in range(12):
                out = cb.step(cue, error=(1.0 if 4 <= k < 8 else 0.0))
        return float(np.mean(out["purkinje"]))

    trained = run(0.02)
    untrained = run(0.0)
    assert trained < untrained * 0.95, (trained, untrained)


# -------------------------------------------------------------- hypothalamus

def test_hypothalamus_drives_are_relieved_by_their_own_action():
    h = Hypothalamus(seed=0)
    for _ in range(3000):
        h.energy = max(0.0, h.energy - 0.0005)
        h.step()
    hungry = h.hunger
    h.eat(0.6)
    for _ in range(500):
        h.step()
    assert h.hunger < hungry, "eating did not reduce hunger"

    h2 = Hypothalamus(seed=0)
    for _ in range(3000):
        h2.step()
    before = h2.fatigue
    for _ in range(800):
        h2.rest(0.01)
        h2.step()
    assert h2.fatigue < before, "rest did not reduce fatigue"


# ------------------------------------------------------- neuromodulators

def test_dopamine_tracks_prediction_error_not_reward():
    m = Neuromodulators(seed=0)
    first = m.reward(1.0)
    for _ in range(60):
        m.reward(1.0)
        m.step()
    late = m.reward(1.0)
    omission = m.reward(0.0)
    assert first > late + 0.2, "dopamine did not habituate to a predicted reward"
    assert omission < 0, "omission produced no negative prediction error"


# --------------------------------------------------------------- prefrontal

def test_prefrontal_holds_a_pattern_after_the_input_stops():
    pf = Prefrontal(120, seed=0)
    pat = np.zeros(120, np.float32)
    pat[10:20] = 1.0
    pf.write(pat, lr=0.4)
    for _ in range(40):
        pf.step(pat, gate=1.0)
    held = pf.state.copy()
    for _ in range(60):
        pf.step(None, gate=0.0)
    assert float(pf.state[10:20].mean()) > 0.2, "activity did not persist"
    assert np.corrcoef(held, pf.state)[0, 1] > 0.7


# -------------------------------------------------------------- whole brain

def test_whole_brain_runs_a_closed_loop():
    brain = WholeBrain(seed=0)
    body = Body(seed=0)
    s = body.reset()
    for _ in range(400):
        out = brain.step(s)
        s, r, done, info = body.act(out["action"])
        if r != 0:
            brain.reinforce(r)
        if done:
            s = body.reset()
    assert 0 <= brain.last_action < 6
    rep = brain.report()
    assert rep["steps"] == 400
    for key in ("l4", "l23", "l5", "thal"):
        assert np.isfinite(rep["rates"][key])


def test_lesion_bypasses_without_breaking_shapes():
    for name in ("thalamus", "basal_ganglia", "hippocampus", "amygdala",
                 "hypothalamus", "cerebellum", "neuromodulators"):
        brain = WholeBrain(seed=0, lesion=(name,))
        out = brain.step(np.zeros(16, np.float32))
        assert 0 <= out["action"] < 6
        assert np.isfinite(out["evidence"]).all()


def test_unknown_lesion_is_rejected():
    with pytest.raises(ValueError):
        WholeBrain(lesion=("cerebellum_",))


def test_basal_ganglia_lesion_opens_every_channel():
    brain = WholeBrain(seed=0, lesion=("basal_ganglia",))
    for _ in range(50):
        brain.step(np.ones(16, np.float32) * 0.2)
    assert (brain.last_gpi < 1e-6).all()
