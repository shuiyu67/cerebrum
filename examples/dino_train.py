"""Train the pool to play the dino game, then render a video of the result.

    python -m examples.dino_train --minutes 12

The readout is trained by PPO from the survival reward alone — there is no
teacher and no imitation. That is the difference between this and the inverted
pendulum, where a PD controller was fitted and the pool could never exceed it.

The action is one continuous number; the dino jumps when it is positive. A
categorical policy would be tidier, but a single-head Gaussian keeps the PPO
implementation shared with the arm task and works because the decision is
essentially "is this the moment or not".
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from cerebrum.control.arm import ArmController
from cerebrum.control.dino import DinoGame, heuristic_jump
from cerebrum.control.ppo import NumpyPPO

N_BINS = 256
STEPS_PER_ACTION = 4


def evaluate(ctrl, policy, episodes: int = 8, max_frames: int = 3000,
             seed0: int = 1000) -> dict:
    scores = []
    for i in range(episodes):
        game = DinoGame(seed=seed0 + i, max_frames=max_frames)
        obs = game.reset()
        while True:
            mu, _ = policy.distribution(ctrl.features(obs))
            obs, _r, done = game.step(bool(mu[0] > 0.0))
            if done:
                break
        scores.append(game.score)
    s = np.array(scores)
    return {"mean": float(s.mean()), "median": float(np.median(s)),
            "min": int(s.min()), "max": int(s.max()),
            "held": int((s >= max_frames).sum())}


def train(minutes: float = 12.0, n_neurons: int = 2000, batch: int = 1024,
          seed: int = 0, out_dir: str = "data"):
    ctrl = ArmController(n_neurons=n_neurons, fan_out=32, n_bins=N_BINS,
                         n_obs=10, steps_per_action=STEPS_PER_ACTION,
                         w_scale=4.0, ginh=20.0, target_frac=0.0, seed=seed)
    policy = NumpyPPO(n_features=N_BINS, n_actions=1, lr=3e-3,
                      ent_coef=0.01, epochs=8, seed=seed)

    deadline = time.time() + minutes * 60.0
    history = []
    best = {"mean": 0.0, "params": None, "iter": 0}
    it = 0
    while time.time() < deadline:
        S, A, LP, R, D = [], [], [], [], []
        game = DinoGame(seed=10_000 + it, max_frames=3000)
        obs = game.reset()
        while len(S) < batch:
            f = ctrl.features(obs)
            a, logp, _mu = policy.act(f)
            obs_next, r, done = game.step(bool(a[0] > 0.0))
            S.append(f); A.append(a); LP.append(logp)
            R.append(r); D.append(float(done))
            obs = game.reset() if done else obs_next
        stats = policy.update(S, A, LP, R, D)
        it += 1

        if it % 10 == 0:
            ev = evaluate(ctrl, policy)
            ev["iter"] = it
            ev["std"] = stats["std"]
            history.append(ev)
            flag = ""
            if ev["mean"] > best["mean"]:
                best = {"mean": ev["mean"], "iter": it,
                        "params": (policy.W_mu.copy(), policy.log_std.copy())}
                flag = "  <- best"
            print("iter %3d | std %.2f | eval mean %.0f median %.0f max %d held %d/8%s"
                  % (it, stats["std"], ev["mean"], ev["median"], ev["max"],
                     ev["held"], flag), flush=True)

    os.makedirs(out_dir, exist_ok=True)
    if best["params"] is not None:
        policy.W_mu, policy.log_std = best["params"]
    np.savez(os.path.join(out_dir, "dino_policy.npz"),
             W_mu=policy.W_mu, log_std=policy.log_std,
             n_neurons=n_neurons, n_bins=N_BINS,
             steps_per_action=STEPS_PER_ACTION, seed=seed)
    with open(os.path.join(out_dir, "dino_history.json"), "w") as fh:
        json.dump(history, fh)
    final = evaluate(ctrl, policy, episodes=12)
    print("final: %s" % json.dumps(final))
    print("saved %s/dino_policy.npz" % out_dir)
    return ctrl, policy, history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=12.0)
    ap.add_argument("--neurons", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(minutes=args.minutes, n_neurons=args.neurons,
          batch=args.batch, seed=args.seed)
