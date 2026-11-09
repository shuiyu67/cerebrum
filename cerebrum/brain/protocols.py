"""Ablation protocols: one experiment per system, each with a control.

An architecture diagram proves nothing. What proves something is removing one
part, holding everything else fixed, and measuring the difference.

Every function here follows the same shape: build the system, measure a
quantity with the mechanism intact, disable exactly that mechanism, measure
again. The seed is the same in both arms, so the comparison is paired.

Run all of them with::

    python -m cerebrum.brain.protocols
"""

from __future__ import annotations

import numpy as np

from .basal_ganglia import BasalGanglia
from .cerebellum import Cerebellum
from .cortex import Cortex
from .hippocampus import Hippocampus
from .neuromodulation import Neuromodulators
from .sheet import NeuronSheet, build_projection
from .thalamus import Thalamus
from .whole_brain import WholeBrain
from .body import Body


# ---------------------------------------------------------------- cortex map

def protocol_self_organisation(epochs: int = 40) -> dict:
    """Can the cortex partition itself? The answer is: only with a neighbourhood.

    Two input families that share no features (one lives in the first half of
    the vector, the other in the second). No region labels are given anywhere.
    The measure is how often two spatially adjacent cells prefer the same
    family — 0.50 is chance, 1.00 is a clean territory.
    """
    rng = np.random.default_rng(1)
    n_in = 64
    a = [np.r_[rng.random(32) > 0.5, np.zeros(32, bool)].astype(np.float32)
         for _ in range(4)]
    b = [np.r_[np.zeros(32, bool), rng.random(32) > 0.5].astype(np.float32)
         for _ in range(4)]
    patterns = a + b

    def run(sigma: float) -> dict:
        sheet = NeuronSheet("ctx", 900, seed=3)
        aff = rng.normal(0, 0.1, (900, n_in)).astype(np.float32)
        aff /= np.linalg.norm(aff, axis=1, keepdims=True) + 1e-9
        for _ in range(epochs):
            for p in patterns:
                sheet.learn_map(aff, p, lr=0.05, sigma=sigma)
        pref = (aff[:, :32].sum(1) > aff[:, 32:].sum(1)).astype(int)
        grid = pref.reshape(30, 30)
        return {
            "adjacency": float((np.mean(grid[:, :-1] == grid[:, 1:])
                                + np.mean(grid[:-1, :] == grid[1:, :])) / 2),
            "n_blocks": int(_count_blocks(grid)),
        }

    winner = run(0.0)      # only the best-matching cell learns (control)
    local = run(0.25)      # winner plus a Gaussian neighbourhood
    return {"control_no_neighbourhood": winner, "with_neighbourhood": local,
            "chance": 0.5}


def _count_blocks(grid: np.ndarray) -> int:
    """Connected components of equal preference (4-neighbourhood)."""
    seen = np.zeros(grid.shape, bool)
    h, w = grid.shape
    count = 0
    for i in range(h):
        for j in range(w):
            if seen[i, j]:
                continue
            count += 1
            stack = [(i, j)]
            seen[i, j] = True
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] \
                            and grid[ny, nx] == grid[i, j]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
    return count


# ------------------------------------------------------------------ thalamus

def protocol_thalamic_gating(steps: int = 400) -> dict:
    """Does the reticular nucleus actually suppress what the cortex ignores?

    A target on channels 0-1 and a distractor on channels 8-9, equal strength.
    The only thing that changes between arms is where the descending cortical
    feedback points. If the thalamus is a passive relay the two arms are
    identical.
    """
    def run(fb_channels, trn_gain=0.6):
        th = Thalamus(16, 10, seed=0, trn_gain=trn_gain)
        sense = np.zeros(16, np.float32)
        sense[[0, 1, 8, 9]] = 1.0
        fb = np.zeros(16, np.float32)
        fb[list(fb_channels)] = 1.0
        relay = np.zeros(16, np.float32)
        for _ in range(steps):
            relay = 0.9 * relay + 0.1 * th.step(sense, fb)["relay"]
        return {"target": float(relay[[0, 1]].mean()),
                "distractor": float(relay[[8, 9]].mean())}

    base = run([])
    on_target = run([0, 1])
    on_distractor = run([8, 9])
    no_trn = run([8, 9], trn_gain=0.0)

    def keep(side, ref):
        return 100.0 * side / (ref + 1e-12)

    return {
        "baseline_relay": base,
        "feedback_on_distractor": {
            "distractor_kept_pct": keep(on_distractor["distractor"],
                                        base["distractor"]),
            "target_kept_pct": keep(on_distractor["target"], base["target"]),
        },
        "feedback_on_target": {
            "target_kept_pct": keep(on_target["target"], base["target"]),
        },
        "reticular_lesioned": {
            "distractor_kept_pct": keep(no_trn["distractor"], base["distractor"]),
        },
    }


# ------------------------------------------------------------ basal ganglia

def protocol_action_selection(steps: int = 60) -> dict:
    """Does the basal ganglia turn graded evidence into a single release?"""
    bg = BasalGanglia(6, seed=0)
    rng = np.random.default_rng(0)

    def run(dopamine: float, lesion: bool) -> dict:
        released, opened, correct = [], [], []
        for _ in range(steps):
            c = np.full(6, 0.18)
            best = int(rng.integers(0, 6))
            c[best] = 0.65
            if lesion:
                gpi = np.zeros(6)
            else:
                # A fresh striatum per trial, settled for a few milliseconds:
                # selection is a transient, and the indirect pathway starts
                # closing the gate again after ~10 ms by design.
                bg = BasalGanglia(6, seed=0)
                for _ in range(6):
                    gpi = bg.step(c, dopamine=dopamine)
            rel = 1.0 - gpi
            released.append(float(rel.max()))
            opened.append(float((rel > 0.5).sum()))
            correct.append(1.0 if int(np.argmax(rel)) == best else 0.0)
        return {"max_release": float(np.mean(released)),
                "open_channels": float(np.mean(opened)),
                "picks_strongest": float(np.mean(correct))}

    intact = run(0.0, lesion=False)
    lesioned = run(0.0, lesion=True)
    high_da = run(1.0, lesion=False)
    low_da = run(-1.0, lesion=False)
    return {"intact": intact, "lesioned": lesioned,
            "da_high": high_da, "da_low": low_da}


# --------------------------------------------------------------- hippocampus

def protocol_pattern_completion(trials: int = 12) -> dict:
    """Store patterns once, then cue with half of one. Does the whole come back?"""
    rng = np.random.default_rng(4)
    n = 400
    patterns = [(rng.random(n) > 0.5).astype(np.float32) for _ in range(trials)]

    def run(store: bool) -> dict:
        hip = Hippocampus(n_cortex=n, n_ec=300, n_dg=1500, n_ca3=800, seed=0)
        if store:
            for p in patterns:
                hip.experience(p)
        overlaps = []
        for p in patterns[:6]:
            cue = p.copy()
            keep = rng.random(n) < 0.5
            cue[keep] = 0.0
            rec = hip.recall(cue, steps=20)
            _, _, full = hip._ca3_code(p)
            done = rec["code"]
            inter = float(np.minimum(done, full).sum())
            union = float(np.maximum(done, full).sum()) + 1e-9
            overlaps.append(inter / union)
        return {"completion": float(np.mean(overlaps))}

    return {"with_storage": run(True), "without_storage": run(False)}


# ------------------------------------------------------------------ amygdala

def protocol_one_trial_fear() -> dict:
    """One pairing, then test the cue alone. And a control with arousal zero."""
    rng = np.random.default_rng(7)
    cs = np.zeros(16, np.float32)
    cs[[0, 1]] = 1.0
    other = np.zeros(16, np.float32)
    other[[8, 9]] = 1.0

    def run(arousal: float) -> float:
        from .amygdala import Amygdala
        am = Amygdala(16, seed=0)
        # one pairing only
        am.learn(cs, us=1.0, arousal=arousal, ach=1.0)
        # test the conditioned cue, and an unpaired cue as a control
        f_cs = _fear_response(am, cs)
        f_other = _fear_response(am, other)
        return {"to_cs": f_cs, "to_other": f_other, "specificity": f_cs - f_other}

    return {"normal_arousal": run(1.0), "arousal_blocked": run(0.0)}


def _fear_response(am, cue, steps: int = 40) -> float:
    """Mean fear over the second half of the response.

    Peak is the wrong statistic here: the previous trial's fear is still
    decaying at the start of the next one, so a peak picks up the previous
    answer. Measured, the unpaired cue scored 1.29 against 1.44 for the
    paired one — both saturated, and the difference invisible.
    """
    am.reset()
    am.fear = 0.0
    vals = []
    for _ in range(steps):
        out = am.step(cue)
        vals.append(out["fear"])
    return float(np.mean(vals[len(vals) // 2:]))


# ---------------------------------------------------------------- cerebellum

def protocol_cerebellum_timing(trials: int = 40) -> dict:
    """Paired cue and error: does the error fall with training, and only then?"""
    rng = np.random.default_rng(9)
    cue = np.zeros(64, np.float32)
    cue[:16] = 1.0

    def run(ltd: float) -> list:
        cb = Cerebellum(64, n_granule=400, n_purkinje=48, n_deep=48,
                        ltd=ltd, seed=0)
        series = []
        for t in range(trials):
            cb.reset()
            err = 1.0
            for k in range(12):
                # the error arrives 4 ms after the cue onset, every trial
                e = err if 4 <= k < 8 else 0.0
                out = cb.step(cue, error=e)
            series.append(float(np.mean(out["purkinje"])))
        return series

    trained = run(0.02)
    untrained = run(0.0)
    return {
        "early_purkinje": float(np.mean(trained[:5])),
        "late_purkinje": float(np.mean(trained[-5:])),
        "no_ltd_early": float(np.mean(untrained[:5])),
        "no_ltd_late": float(np.mean(untrained[-5:])),
    }


# ------------------------------------------------------------ neuromodulators

def protocol_reward_prediction() -> dict:
    """Dopamine should follow the *error*, not the reward."""
    m = Neuromodulators(seed=0)
    out = []
    for i in range(40):
        rpe = m.reward(1.0)
        m.step()
        out.append(rpe)
    # then omit the reward entirely
    omitted = []
    for i in range(10):
        rpe = m.reward(0.0)
        m.step()
        omitted.append(rpe)
    return {"first_reward_rpe": float(out[0]),
            "last_reward_rpe": float(out[-1]),
            "first_omission_rpe": float(omitted[0]),
            "value": float(m.value)}


# ------------------------------------------------------------------ assembly

def protocol_lesion_suite(steps: int = 6000, seeds: tuple = (0,)) -> dict:
    """Lesion each system and measure survival behaviour.

    This is the only experiment that involves the whole loop, and it is the
    one that says whether the parts matter when they are all running.
    """
    systems = ("thalamus", "basal_ganglia", "hippocampus", "amygdala",
               "hypothalamus", "cerebellum", "neuromodulators")
    def run(lesion: tuple) -> dict:
        eaten, caught, reward = [], [], []
        for sd in seeds:
            b = WholeBrain(seed=sd, lesion=lesion)
            body = Body(seed=sd)
            s = body.reset()
            total = 0.0
            for _ in range(steps):
                # the predator's bite is the unconditioned stimulus; without
                # this the amygdala never learns and its lesion is a no-op
                out = b.step(s, us=1.0 if body.fear > 0.85 else 0.0)
                s, r, done, info = body.act(out["action"])
                if r != 0:
                    b.reinforce(r)
                total += r
                if done:
                    s = body.reset()
            eaten.append(body.eaten)
            caught.append(body.caught)
            reward.append(total)
        e, c = float(np.mean(eaten)), float(np.mean(caught))
        return {"eaten": e, "caught": c,
                "reward": float(np.mean(reward)),
                # Both counts fall together when a lesion simply makes the
                # animal move less, so the ratio is the measure that separates
                # "better at finding food" from "too still to meet anything".
                "eaten_per_encounter": e / (e + c) if (e + c) else 0.0}

    result = {"intact": run(())}
    for name in systems:
        result[f"without_{name}"] = run((name,))
    return result


# --------------------------------------------------------------------- main

ALL = {
    "self_organisation": protocol_self_organisation,
    "thalamic_gating": protocol_thalamic_gating,
    "action_selection": protocol_action_selection,
    "pattern_completion": protocol_pattern_completion,
    "one_trial_fear": protocol_one_trial_fear,
    "cerebellum_timing": protocol_cerebellum_timing,
    "reward_prediction": protocol_reward_prediction,
}


def _fmt(v):
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}={_fmt(x)}" for k, x in v.items()) + "}"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main(argv=None) -> int:
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    names = [a for a in args if a in ALL or a == "lesions"]
    if not names:
        names = list(ALL) + ["lesions"]
    for name in names:
        print(f"\n=== {name} ===")
        if name == "lesions":
            out = protocol_lesion_suite()
        else:
            out = ALL[name]()
        for k, v in out.items():
            print(f"  {k}: {_fmt(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
