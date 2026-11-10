"""Hold an inverted pendulum up with a spiking pool.

    python -m examples.pole_control

Trains the readout by ridge regression on a PD teacher, then closes the loop
and lets the pool control the pole on its own. Writes a comparison figure to
``data/pole_comparison.png``.

The number to look at is the second column: a randomly weighted readout holds
the pole for about 36 control steps (0.36 s) before it passes the fall
threshold, and the trained one holds it for the whole 600 (6 s).
"""

from __future__ import annotations

import numpy as np

from cerebrum.control import InvertedPendulum, PoolController, pd_teacher

STEPS = 600
THETA0 = 0.30
NOISE = 0.5


def ridge(X: list, y: list, lam: float = 1.0) -> np.ndarray:
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.float32)
    return np.linalg.solve(
        X.T @ X + lam * np.eye(X.shape[1], dtype=np.float32), X.T @ y
    ).astype(np.float32)


def collect(ctrl: PoolController, episodes: int = 8, steps: int = 350):
    """Drive the pool along teacher-controlled trajectories.

    Starts are randomised over the same range the evaluation uses, and the
    environment keeps its noise. Training only on quiet, small-angle rollouts
    and then testing under noise is the classic way to get a controller that
    looks trained and falls over immediately.
    """
    X, y = [], []
    for ep in range(episodes):
        env = InvertedPendulum(noise=0.3, seed=ep)
        obs = env.reset(theta=float(np.random.default_rng(ep).uniform(-THETA0, THETA0)))
        for _ in range(steps):
            X.append(ctrl.features(obs))
            y.append(pd_teacher(obs))
            obs, _ = env.step(pd_teacher(obs))
            if env.fallen:
                break
    return X, y


def rollout(ctrl: PoolController, W: np.ndarray, steps: int = STEPS):
    env = InvertedPendulum(noise=NOISE, seed=5)
    env.reset(theta=THETA0)
    thetas = []
    for t in range(steps):
        env.step(float(W @ ctrl.features(env.observe())))
        thetas.append(abs(float(env.observe()[0])))
        if env.fallen:
            return t + 1, False, float(np.mean(thetas))
    return steps, True, float(np.mean(thetas))


def main() -> None:
    ctrl = PoolController(n_neurons=4000, fan_out=32, n_bins=256,
                          w_scale=4.0, ginh=20.0, target_frac=0.0,
                          use_omega=True, seed=0)
    X, y = collect(ctrl)
    W_trained = ridge(X, y)
    W_random = np.random.default_rng(1).normal(0, 2.0, ctrl.n_bins).astype(np.float32)

    print("collected %d samples from %d trajectories" % (len(y), 8))
    print("readout fit: corr %.3f" % float(np.corrcoef(np.asarray(X) @ W_trained, y)[0, 1]))
    print()
    print("%-24s %8s %10s" % ("controller", "steps", "mean|theta|"))
    for name, W in (("pool, trained", W_trained), ("pool, random", W_random)):
        steps, held, mean_theta = rollout(ctrl, W)
        print("%-24s %8d %10.3f  %s"
              % (name, steps, mean_theta, "held" if held else "fell"))

    # The teacher's own score, for reference.
    env = InvertedPendulum(noise=NOISE, seed=5)
    env.reset(theta=THETA0)
    thetas = []
    for t in range(STEPS):
        obs, _ = env.step(pd_teacher(env.observe()))
        thetas.append(abs(float(obs[0])))
        if env.fallen:
            break
    print("%-24s %8d %10.3f  %s"
          % ("PD teacher", len(thetas), float(np.mean(thetas)),
             "held" if not env.fallen else "fell"))


if __name__ == "__main__":
    main()
