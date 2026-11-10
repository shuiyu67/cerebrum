"""Learning and acting at the same time.

Every earlier experiment in this repository had a training phase and then a
frozen phase -- fit the readout on a batch, freeze the pool, then measure. That
is not how cortex works. Here one loop does both: read the observation, fire,
predict time-to-impact, compare the prediction with the time-to-impact that is
*already in the observation*, and let that comparison move the readout and --
through three-factor plasticity -- the synapses, on that frame, before the next
one arrives.

There is no teacher in this file. ``obs[2]`` is one of the eleven numbers the
pool is given, so the error is the gap between what the readout predicts from
the pool state and what the same frame says about itself. That is prediction
coding, not supervision.

Two things had to be right for this to work at all, and both were found by
measuring rather than by reasoning:

**The gate must be signed.** Using ``-abs(error)`` as the reward is a constant
negative push: every active synapse is weakened every frame, which is a global
inhibitory clamp rather than a learning rule. The pool collapses to the
"never jump" score of 117 within one episode. The signed error balances,
because predicting too early and too late pull in opposite directions.

**The gate must be a *change* in error, not the error.** A single-frame
prediction error says how good the readout is, not whether the pool's last few
milliseconds helped. Feeding it to the trace credits synapses for something
they did not do. The temporal difference ``|e_{t-1}| - |e_t|`` -- did the
prediction get better than a moment ago -- is what turns the pool's online
plasticity from harmful (146 frames) into useful (444).

Measured over three seeds, fourteen episodes each, weights persisting across
episodes and membrane state reset:

    pool frozen, readout online        late-episode mean 310
    pool + readout online              late-episode mean 444   (+43%, 3/3 seeds)
    best single episode                frozen 1255, online 1666

Both conditions decline from early to late episodes -- the readout at
learning rate 0.01 drifts, and drift is the next problem rather than a solved
one. The pool's plasticity pushes back against that drift instead of causing
it. That is the first time in this repository that enabling plasticity has
improved anything in a closed loop.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from cerebrum.control.arm import ArmController
from cerebrum.control.dino import DinoGame

# The biological operating point established earlier in the README: 28 Hz at
# fan-out 128 keeps 95% of the code quality at a third of the spikes, and it is
# the only configuration where three-factor plasticity has ever measured > 0.
FAN, W_SCALE, GINH = 128, 2.0, 400.0
N_NEURONS, N_BINS, N_OBS, SPA = 4000, 256, 11, 4
THRESH, SMOOTH = 13.0 / 60.0, 0.4

# Gain on the pool's plasticity. Measured: 0.05 and 0.25 drift too little to
# matter (late 398/399), 250 destroys the pool (late 117), 1.0 is the peak.
POOL_GAIN = 1.0
READOUT_LR = 0.01


def build(pool_gain: float, init: str):
    ctrl = ArmController(n_neurons=N_NEURONS, fan_out=FAN, n_bins=N_BINS,
                         n_obs=N_OBS, steps_per_action=SPA, w_scale=W_SCALE,
                         ginh=GINH, target_frac=0.0,
                         stdp=0.01 if pool_gain > 0 else 0.0,
                         three_factor=pool_gain, seed=0)
    d = np.load(init)
    ctrl.pool.w[:] = d["w"]
    return ctrl, d["W_mu"], float(d["bias"])


def lifetime(pool_gain: float, readout: bool, init: str, seed0: int,
             episodes: int, cap: int):
    """One continuous lifetime. Weights persist; membrane state does not."""
    ctrl, W0, b0 = build(pool_gain, init)
    w0 = ctrl.pool.w.copy()
    W, b = W0.copy(), b0
    scores, r2 = [], []
    for ep in range(episodes):
        ctrl.pool.reset()
        game = DinoGame(seed=seed0 + ep, max_frames=cap)
        obs = game.reset()
        sm, prev, preds, trues = 1.0, None, [], []
        while True:
            f = ctrl.features(obs)
            tti = float(obs[2])
            pred = float(W @ f) + b
            err = pred - tti

            if readout:
                # Online least mean squares. The feature vector has unit norm,
                # so this is a normalised step and lr is a real learning rate.
                W -= np.float32(READOUT_LR * err) * f
                b -= READOUT_LR * err * 0.05

            a = abs(err)
            if pool_gain > 0.0 and prev is not None:
                # Did the prediction get better than a moment ago? That, not
                # the error itself, is what the synapse should be credited for.
                ctrl.pool.apply_reward(prev - a)
            prev = a

            sm = SMOOTH * sm + (1.0 - SMOOTH) * pred
            obs, _r, done = game.step((obs[-1] < 0.5) and (sm < THRESH))
            preds.append(pred)
            trues.append(tti)
            if done:
                break

        scores.append(game.score)
        p, t = np.array(preds), np.array(trues)
        r2.append(float(1.0 - ((p - t) ** 2).sum() /
                        max(1e-9, ((t - t.mean()) ** 2).sum())))
    return (np.array(scores), np.array(r2),
            float(np.abs(ctrl.pool.w - w0).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=14)
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--seeds", default="13000,14000,15000")
    ap.add_argument("--init", default="data/dino_learned_full.npz")
    ap.add_argument("--out", default="data/dino_online.json")
    ap.add_argument("--plot", default="data/dino_online.png")
    args = ap.parse_args()

    seed0s = [int(s) for s in args.seeds.split(",")]
    cases = [("frozen", 0.0, False), ("readout online", 0.0, True),
             ("readout + pool online", POOL_GAIN, True)]

    out = {}
    for name, gain, ro in cases:
        runs = []
        for s in seed0s:
            sc, r2, drift = lifetime(gain, ro, args.init, s,
                                     args.episodes, args.cap)
            runs.append((sc, r2, drift))
            print("%-24s seed %d | mean %4.0f max %4d | R2 %.3f | drift %.3f"
                  % (name, s, sc.mean(), sc.max(), r2.mean(), drift), flush=True)
        late = [float(sc[len(sc) // 2:].mean()) for sc, _, _ in runs]
        early = [float(sc[:len(sc) // 2].mean()) for sc, _, _ in runs]
        out[name] = dict(scores=[sc.tolist() for sc, _, _ in runs],
                         early=float(np.mean(early)), late=float(np.mean(late)),
                         late_per_seed=late,
                         best=int(max(sc.max() for sc, _, _ in runs)))
        print("  -> early %.0f  late %.0f  best %d  late per seed %s\n"
              % (out[name]["early"], out[name]["late"], out[name]["best"],
                 " ".join("%.0f" % v for v in late)), flush=True)

    print("%-24s %8s %8s %8s" % ("condition", "early", "late", "best"))
    for name, d in out.items():
        print("%-24s %8.0f %8.0f %8d"
              % (name, d["early"], d["late"], d["best"]))

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), facecolor="#0d1117")
        for a in (ax, ax2):
            a.set_facecolor("#0d1117")
            for s in a.spines.values():
                s.set_color("#30363d")
            a.tick_params(colors="#8b949e")
            a.grid(True, color="#21262d", lw=0.6)
            a.set_xlabel("episode (same lifetime, no reset)", color="#8b949e",
                         fontsize=9)
            a.set_ylabel("frames survived", color="#8b949e", fontsize=9)
            a.set_title("learning while acting", color="#e6edf3", fontsize=10)
        colors = {"frozen": "#8b949e", "readout online": "#58a6ff",
                  "readout + pool online": "#3fb950"}
        for name, d in out.items():
            for run in d["scores"]:
                r = np.array(run, float)
                ax.plot(np.arange(len(r)), r, color=colors[name], lw=1.0,
                        alpha=0.35)
            m = np.mean([run for run in d["scores"]], axis=0)
            ax2.plot(np.arange(len(m)), m, color=colors[name], lw=2.0,
                     marker="o", ms=3, label=name)
        ax.set_title("per episode, three seeds", color="#e6edf3", fontsize=10)
        ax2.legend(facecolor="#161b22", edgecolor="#30363d",
                   labelcolor="#e6edf3", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140, facecolor="#0d1117")
        print("wrote", args.plot)
    except Exception as exc:  # plotting is optional
        print("plot skipped:", exc)

    print("wrote", args.out)


if __name__ == "__main__":
    main()
