"""Does online plasticity buy adaptation, or only performance?

The previous experiment showed the plastic pool holds its score while the
frozen one declines. Holding could be two different things: a system that
learns the statistics of its world, or a system whose synapses happen to
average out to something that decays more slowly.

This separates them with a step change. Both pools live through an identical
first phase at the shipped difficulty, then the scroll speed ramps four times
faster than before -- the reaction window halves, and the policy that was
adequate a moment ago is not. A system that learned the world should recover
faster than one that did not, because the thing it learned is still the thing
that changed.

The measured quantity is the recovery curve after the switch, not the level
before it.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from cerebrum.control.arm import ArmController
from cerebrum.control.dino import DinoGame

FAN, W_SCALE, GINH = 128, 2.0, 400.0
N_NEURONS, N_BINS, N_OBS, SPA = 4000, 256, 11, 4
THRESH, SMOOTH = 13.0 / 60.0, 0.4
POOL_GAIN, READOUT_LR = 1.0, 0.01

NORMAL_RAMP = 600.0   # shipped: speed 5 -> 10 over 3000 frames
SHIFT_RAMP = 150.0    # same ceiling reached in 750 frames


def lifetime(pool_gain: float, seed0: int, n_before: int, n_after: int,
             cap: int, init: str):
    ctrl_d = np.load(init)
    ctrl = ArmController(n_neurons=N_NEURONS, fan_out=FAN, n_bins=N_BINS,
                         n_obs=N_OBS, steps_per_action=SPA, w_scale=W_SCALE,
                         ginh=GINH, target_frac=0.0,
                         stdp=0.01 if pool_gain > 0 else 0.0,
                         three_factor=pool_gain, seed=0)
    ctrl.pool.w[:] = ctrl_d["w"]
    w0 = ctrl.pool.w.copy()
    W, b = ctrl_d["W_mu"].copy(), float(ctrl_d["bias"])

    before, after = [], []
    for ep in range(n_before + n_after):
        shifted = ep >= n_before
        ctrl.pool.reset()
        game = DinoGame(seed=seed0 + ep, max_frames=cap)
        game.speed_ramp = SHIFT_RAMP if shifted else NORMAL_RAMP
        obs = game.reset()
        sm, prev = 1.0, None
        while True:
            f = ctrl.features(obs)
            pred = float(W @ f) + b
            err = pred - float(obs[2])
            W -= np.float32(READOUT_LR * err) * f
            b -= READOUT_LR * err * 0.05
            a = abs(err)
            if pool_gain > 0.0 and prev is not None:
                ctrl.pool.apply_reward(prev - a)
            prev = a
            sm = SMOOTH * sm + (1.0 - SMOOTH) * pred
            obs, _r, done = game.step((obs[-1] < 0.5) and (sm < THRESH))
            if done:
                break
        (after if shifted else before).append(game.score)

    return (np.array(before), np.array(after),
            float(np.abs(ctrl.pool.w - w0).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=int, default=10)
    ap.add_argument("--after", type=int, default=10)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--seeds", default="21000,22000")
    ap.add_argument("--init", default="data/dino_learned_full.npz")
    ap.add_argument("--out", default="data/dino_shift.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    out = {}
    for name, gain in (("frozen", 0.0), ("plastic", POOL_GAIN)):
        before_all, after_all = [], []
        for s in seeds:
            bb, aa, drift = lifetime(gain, s, args.before, args.after,
                                     args.cap, args.init)
            before_all.append(bb)
            after_all.append(aa)
            print("%-8s seed %d | before %4.0f | after %4.0f | first3 %s | last3 %s | drift %.3f"
                  % (name, s, bb.mean(), aa.mean(),
                     " ".join("%4d" % v for v in aa[:3]),
                     " ".join("%4d" % v for v in aa[-3:]), drift), flush=True)
        out[name] = dict(before=[r.tolist() for r in before_all],
                         after=[r.tolist() for r in after_all],
                         before_mean=float(np.mean([r.mean() for r in before_all])),
                         after_mean=float(np.mean([r.mean() for r in after_all])),
                         drop=float(np.mean([bb.mean() for bb in before_all])
                                    - np.mean([aa.mean() for aa in after_all])))
        print("  -> before %.0f after %.0f drop %.0f\n"
              % (out[name]["before_mean"], out[name]["after_mean"],
                 out[name]["drop"]), flush=True)

    print("%-10s %8s %8s %8s" % ("condition", "before", "after", "drop"))
    for name, d in out.items():
        print("%-10s %8.0f %8.0f %8.0f"
              % (name, d["before_mean"], d["after_mean"], d["drop"]))

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        for s in ax.spines.values():
            s.set_color("#30363d")
        ax.tick_params(colors="#8b949e")
        ax.grid(True, color="#21262d", lw=0.6)
        colors = {"frozen": "#8b949e", "plastic": "#3fb950"}
        for name, d in out.items():
            allr = [np.array(r, float) for r in d["before"] + d["after"]]
            # before and after are separate lists; concatenate per seed
            for i in range(len(d["before"])):
                r = np.concatenate([np.array(d["before"][i], float),
                                    np.array(d["after"][i], float)])
                ax.plot(np.arange(len(r)), r, color=colors[name], lw=1.0,
                        alpha=0.45)
            n = args.before
            m = np.concatenate([np.mean([np.array(x, float) for x in d["before"]], axis=0),
                                np.mean([np.array(x, float) for x in d["after"]], axis=0)])
            ax.plot(np.arange(len(m)), m, color=colors[name], lw=2.2,
                    marker="o", ms=3, label=name)
        ax.axvline(args.before - 0.5, color="#f0883e", ls="--", lw=1.2)
        ax.text(args.before - 0.3, ax.get_ylim()[1] * 0.92,
                " speed ramp x4", color="#f0883e", fontsize=9)
        ax.set_xlabel("episode", color="#8b949e", fontsize=9)
        ax.set_ylabel("frames survived", color="#8b949e", fontsize=9)
        ax.set_title("adaptation after an abrupt change in the world",
                     color="#e6edf3", fontsize=10)
        ax.legend(facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#e6edf3", fontsize=8)
        fig.tight_layout()
        fig.savefig("data/dino_shift.png", dpi=140, facecolor="#0d1117")
        print("wrote data/dino_shift.png")
    except Exception as exc:
        print("plot skipped:", exc)

    print("wrote", args.out)


if __name__ == "__main__":
    main()
