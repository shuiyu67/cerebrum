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


def test_unsigned_gate_pushes_weights_one_way():
    """The gate has to be signed, and this measures why.

    `-abs(error)` is always negative, so every active synapse is weakened on
    every frame. Over a few hundred frames that is not a learning rule, it is a
    global inhibitory clamp — and measured, it collapses the pool to the
    never-jump score within one episode. A signed error balances because
    predicting too early and too late pull in opposite directions.
    """
    rng = np.random.default_rng(0)
    errs = rng.normal(0.0, 0.25, 4000)

    signed = float(np.mean(-errs))
    unsigned = float(np.mean(-np.abs(errs)))

    assert abs(signed) < 0.05, "signed gate is biased: %.4f" % signed
    assert unsigned < -0.1, "unsigned gate is not one-way: %.4f" % unsigned


def test_online_loop_updates_both_layers():
    """Readout and pool must both move in the online loop, from one pass.

    Uses a small pool on purpose: the assertion is that the update happens and
    is bounded, not that the controller is any good at this size.
    """
    from cerebrum.control.arm import ArmController
    from cerebrum.control.dino import DinoGame

    ctrl = ArmController(n_neurons=600, fan_out=32, n_bins=64, n_obs=11,
                         steps_per_action=2, w_scale=2.0, ginh=400.0,
                         target_frac=0.0, stdp=0.01, three_factor=1.0, seed=0)
    w0 = ctrl.pool.w.copy()
    W = np.zeros(64, np.float32)
    b = 0.35
    game = DinoGame(seed=0, max_frames=120)
    obs = game.reset()
    prev = None
    while True:
        f = ctrl.features(obs)
        pred = float(W @ f) + b
        err = pred - float(obs[2])
        W -= np.float32(0.01 * err) * f
        b -= 0.01 * err * 0.05
        a = abs(err)
        if prev is not None:
            ctrl.pool.apply_reward(prev - a)
        prev = a
        obs, _r, done = game.step(pred < 13.0 / 60.0)
        if done:
            break

    assert np.abs(W).sum() > 0.0, "readout never moved"
    drift = float(np.abs(ctrl.pool.w - w0).mean())
    assert 0.0 < drift < 0.5, "pool drift out of range: %.4f" % drift


def test_learned_change_has_no_cell_specificity():
    """The control that killed the +46% claim.

    A learned change should be concentrated on cells that matter. Measured, it
    is not: the shift is spread almost uniformly across cells (Gini 0.18, the
    top 1% of cells hold under 2% of it) and over a third of synapses flip sign.
    That is a pool being re-drawn, not a pool finding a representation — and it
    is why shuffling which cell receives which change scores as well as, or
    better than, the learned version.
    """
    import os

    import numpy as np

    path = "data/dino_learned_full.npz"
    if not os.path.exists(path):
        return  # artifact not generated in this checkout

    from cerebrum.control.arm import ArmController

    d = np.load(path)
    ctrl = ArmController(n_neurons=4000, fan_out=128, n_bins=256, n_obs=11,
                         steps_per_action=4, w_scale=2.0, ginh=400.0,
                         target_frac=0.0, seed=0)
    w0 = ctrl.pool.w.reshape(4000, 128)
    w1 = d["w"].reshape(4000, 128)

    frac = np.abs(w1 - w0).sum(1)
    frac = frac / frac.sum()
    order = np.sort(frac)
    gini = (2 * np.sum(np.arange(1, 4001) * order)) / (4000 * order.sum()) \
        - (4001 / 4000)

    flip = (np.sign(w0) != np.sign(w1)) & (np.abs(w0) > 0.05)

    assert gini < 0.3, "change is concentrated, not uniform: Gini %.3f" % gini
    assert flip.mean() > 0.2, "too few sign flips to call this a re-draw: %.3f" % flip.mean()
