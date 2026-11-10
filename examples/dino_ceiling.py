"""The gap between "measurably better" and "good enough".

Six rounds of plasticity experiments in this repository ended the same way: the
measurement moved in the right direction by 10-30%, and the behaviour did not
change. This file makes that pattern explicit in one place, on one pool, with
one number attached to each side of it.

The pool is trained with three-factor plasticity at several gains, then three
quantities are measured on held-out seeds with held-out readouts:

    R^2 over all frames              how much of the time-to-impact is in the
                                     state at all
    RMSE inside the decision band    the error where the *choice* is made
                                     (8-18 frames to impact)
    RMSE at the threshold            the same, at 12-14 frames

The result is a clean statement of the wall this architecture sits against: the
error at the decision point falls by about a third, from 4.9 frames to 3.6, and
the jump window is about 9 frames. A controller needs to be right inside a
window of a few frames, so a 28% reduction is real progress that changes
nothing.

That is the general shape of every result in this repository — capacity 0.80 to
1.04 where language needs 10, R^2 0.85 to 0.93 where control needs 0.99 — and
it is why parameter tuning cannot get there. A tuning knob buys tens of percent;
the task is asking for an order of magnitude.
"""

from __future__ import annotations

import argparse

import numpy as np

from cerebrum.control.arm import ArmController
from cerebrum.control.dino import DinoGame, heuristic_jump

FAN, W_SCALE, GINH = 128, 2.0, 400.0
N_NEURONS, N_BINS, N_OBS, SPA = 4000, 256, 11, 4
THRESH = 13.0 / 60.0


def build(gain: float, init: str) -> ArmController:
    ctrl = ArmController(n_neurons=N_NEURONS, fan_out=FAN, n_bins=N_BINS,
                         n_obs=N_OBS, steps_per_action=SPA, w_scale=W_SCALE,
                         ginh=GINH, target_frac=0.0,
                         stdp=0.01 if gain > 0 else 0.0,
                         three_factor=gain, seed=0)
    ctrl.pool.w[:] = np.load(init)["w"]
    return ctrl


def collect(ctrl: ArmController, n: int, seed0: int):
    """States under the heuristic, with 5% exploration so the pool sees states
    its own policy would not visit."""
    ctrl.pool.reset()
    rng = np.random.default_rng(1)
    F, T = [], []
    for k in range(40):
        game = DinoGame(seed=seed0 + k, max_frames=2000)
        obs = game.reset()
        while len(F) < n:
            jump = (heuristic_jump(obs, lead=1.3) if rng.random() > 0.05
                    else bool(rng.random() > 0.5))
            F.append(ctrl.features(obs))
            T.append(obs[2])
            obs, _r, done = game.step(jump)
            if done:
                break
        if len(F) >= n:
            break
    return (np.asarray(F[:n], np.float32), np.asarray(T[:n], np.float32))


def train(ctrl: ArmController, steps: int, init: str, seed0: int = 33000):
    d = np.load(init)
    W, b = d["W_mu"], float(d["bias"])
    game = DinoGame(seed=seed0, max_frames=3000)
    obs = game.reset()
    ctrl.pool.reset()
    prev = None
    for _ in range(steps):
        f = ctrl.features(obs)
        a = abs(float(W @ f + b) - float(obs[2]))
        if prev is not None:
            ctrl.pool.apply_reward(prev - a)
        prev = a
        obs, _r, done = game.step((obs[-1] < 0.5) and (float(W @ f + b) < THRESH))
        if done:
            obs = game.reset()
            ctrl.pool.reset()
            prev = None


def measure(ctrl: ArmController, n: int, seed0: int):
    F, T = collect(ctrl, n, seed0)
    A = np.hstack([F, np.ones((len(F), 1), np.float32)])
    coef, *_ = np.linalg.lstsq(A, T, rcond=None)
    pred = A @ coef
    err = np.abs(pred - T) * 60.0

    band = (T > 8 / 60.0) & (T < 18 / 60.0)
    near = (T > 12 / 60.0) & (T < 14 / 60.0)
    r2 = float(1.0 - ((pred - T) ** 2).sum() / ((T - T.mean()) ** 2).sum())
    return dict(r2=r2, rmse_all=float(err.mean()),
                rmse_band=float(err[band].mean()),
                rmse_near=float(err[near].mean()),
                corr_band=float(np.corrcoef(pred[band], T[band])[0, 1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--gains", default="0,0.3,1,3,10")
    ap.add_argument("--init", default="data/dino_learned_full.npz")
    args = ap.parse_args()

    gains = [float(g) for g in args.gains.split(",")]
    print("%6s | %7s %7s %7s %7s | %8s" %
          ("gain", "R^2", "all", "band", "near", "drift"))
    base_w = np.load(args.init)["w"].copy() if 0.0 in gains else None
    base = None
    for g in gains:
        ctrl = build(g, args.init)
        if g > 0:
            train(ctrl, args.steps, args.init)
        m = measure(ctrl, args.n, 50000)
        drift = (float(np.abs(ctrl.pool.w - base_w).mean())
                 if base_w is not None else 0.0)
        if base is None:
            base = m
        print("%6.1f | %7.4f %7.2f %7.2f %7.2f | %8.4f"
              % (g, m["r2"], m["rmse_all"], m["rmse_band"], m["rmse_near"],
                 drift), flush=True)

    print()
    print("A jump window is about 9 frames wide; the error at the decision")
    print("point has to be a couple of frames for control to be reliable.")
    if base is not None:
        print("best improvement in decision-point error: %.2f -> %.2f frames"
              % (base["rmse_near"], m["rmse_near"]))


if __name__ == "__main__":
    main()
