"""The control task exists to be a yardstick the language model was not.

Its whole value is that the sign of the result is obvious before it is run:
without control the pole falls, with a PD controller it stands, and a random
readout on a spiking pool has no reason to do better than falling. If the
pool's readout cannot beat the random one here, the pool is not carrying a
control signal and the number says so.
"""

import numpy as np

from cerebrum.control import InvertedPendulum, PoolController, pd_teacher


def _rollout(env, policy, steps=400, theta0=0.30):
    env.reset(theta=theta0)
    for t in range(steps):
        env.step(policy(env.observe()))
        if env.fallen:
            return t + 1, False
    return steps, True


def test_pole_falls_without_control():
    env = InvertedPendulum(noise=0.0, seed=0)
    steps, held = _rollout(env, lambda o: 0.0)
    assert not held, "uncontrolled pole stayed up for %d steps" % steps
    assert steps < 100, "uncontrolled pole survived %d steps, threshold too loose" % steps


def test_pd_teacher_holds_the_pole():
    env = InvertedPendulum(noise=0.5, seed=5)
    steps, held = _rollout(env, lambda o: pd_teacher(o))
    assert held, "PD teacher fell at step %d" % steps


def test_pool_controller_produces_a_usable_signal():
    """Shape and finiteness of the state, and that the pool is alive.

    The trained-vs-random comparison lives in examples/pole_control.py because
    it needs a few thousand teacher-controlled steps to fit the readout; this
    test only guarantees the plumbing.
    """
    ctrl = PoolController(n_neurons=2000, fan_out=32, n_bins=128,
                          w_scale=4.0, ginh=20.0, target_frac=0.0, seed=0)
    obs = np.array([0.05, -0.2], np.float32)
    feats = ctrl.features(obs)
    assert feats.shape == (128,)
    assert np.isfinite(feats).all()
    assert abs(float(np.linalg.norm(feats)) - 1.0) < 1e-4, "state is not unit norm"
    assert ctrl.pool.rate_hz() > 0.0, "pool never fired"

    torque = ctrl.act(obs)
    assert np.isfinite(torque)


def test_partial_observability_switch_changes_the_drive():
    """With use_omega off the angle must still drive the pool, otherwise the
    partial-observability comparison measures nothing."""
    full = PoolController(n_neurons=2000, fan_out=32, n_bins=64,
                          w_scale=4.0, ginh=20.0, target_frac=0.0,
                          use_omega=True, seed=0)
    part = PoolController(n_neurons=2000, fan_out=32, n_bins=64,
                          w_scale=4.0, ginh=20.0, target_frac=0.0,
                          use_omega=False, seed=0)
    obs = np.array([0.2, 0.5], np.float32)
    idx_full, _ = full._drive(obs)
    idx_part, _ = part._drive(obs)
    assert idx_part.size > 0, "angle alone drove nothing"
    assert idx_full.size > idx_part.size, "omega channel added no cells"

    # A pure velocity state drives the full controller and nothing in the
    # partial one, which is the point of the switch.
    idx_full2, _ = full._drive(np.array([0.0, 0.9], np.float32))
    idx_part2, _ = part._drive(np.array([0.0, 0.9], np.float32))
    assert idx_full2.size > 0 and idx_part2.size == 0
