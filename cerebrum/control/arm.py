"""A two-joint planar arm the pool has to learn to use.

Why an arm and not another pendulum
-----------------------------------
The pendulum is linearly controllable, so a linear readout fitted on a PD
teacher is already optimal and the pool can only tie. An arm is different in
the way that matters: reaching for a point is a nonlinear problem with
redundant solutions, there is no closed-form teacher to imitate, and the map
from joint torques to fingertip position goes through a matrix that changes
with configuration. A readout that merely copies a teacher cannot do this,
which is the point — the pool has to be trained by reward, not imitation.

The arm lies in a horizontal plane so gravity drops out. That leaves inertia
and the Coriolis terms, which is the standard reaching benchmark and keeps the
dynamics honest without turning the file into a rigid-body textbook.

State is 10 numbers: sin/cos of both joints (angles wrap, so sin/cos keeps the
representation continuous), both angular velocities, the fingertip, and the
target. The sin/cos encoding is not cosmetic — feeding a raw angle makes the
network see a discontinuity at +-pi, so two physically identical poses look
maximally different.
"""

from __future__ import annotations

import numpy as np

from ..lm.reservoir import SparseReservoir


class PlanarArm:
    """Two links, two torque inputs, fingertip position as the output.

    Parameters are those of a light desktop-scale arm: 30 cm links, 1 kg each,
    torque limit high enough to move the joints but not to slam them.
    """

    def __init__(self, l1: float = 0.30, l2: float = 0.30,
                 m1: float = 1.0, m2: float = 1.0, dt: float = 0.01,
                 max_torque: float = 12.0, damping: float = 0.8,
                 seed: int = 0) -> None:
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.m1 = float(m1)
        self.m2 = float(m2)
        self.lc1 = self.l1 / 2.0
        self.lc2 = self.l2 / 2.0
        self.I1 = self.m1 * self.l1 ** 2 / 12.0
        self.I2 = self.m2 * self.l2 ** 2 / 12.0
        self.dt = float(dt)
        self.max_torque = float(max_torque)
        self.damping = float(damping)
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros(2)
        self.dq = np.zeros(2)
        self.target = np.zeros(2)
        self.t = 0
        self.max_reach = self.l1 + self.l2

    # -- geometry --------------------------------------------------------

    def forward(self, q: np.ndarray | None = None) -> np.ndarray:
        q = self.q if q is None else q
        x = self.l1 * np.cos(q[0]) + self.l2 * np.cos(q[0] + q[1])
        y = self.l1 * np.sin(q[0]) + self.l2 * np.sin(q[0] + q[1])
        return np.array([x, y])

    def observe(self) -> np.ndarray:
        tip = self.forward()
        s = self.max_reach
        return np.array([np.sin(self.q[0]), np.cos(self.q[0]),
                         np.sin(self.q[1]), np.cos(self.q[1]),
                         self.dq[0] / 5.0, self.dq[1] / 5.0,
                         tip[0] / s, tip[1] / s,
                         self.target[0] / s, self.target[1] / s], np.float32)

    # -- dynamics --------------------------------------------------------

    def mass_matrix(self) -> np.ndarray:
        c2 = np.cos(self.q[1])
        m11 = (self.m1 * self.lc1 ** 2
               + self.m2 * (self.l1 ** 2 + self.lc2 ** 2
                            + 2.0 * self.l1 * self.lc2 * c2)
               + self.I1 + self.I2)
        m12 = self.m2 * (self.lc2 ** 2 + self.l1 * self.lc2 * c2) + self.I2
        m22 = self.m2 * self.lc2 ** 2 + self.I2
        return np.array([[m11, m12], [m12, m22]])

    def coriolis(self) -> np.ndarray:
        h = self.m2 * self.l1 * self.lc2 * np.sin(self.q[1])
        c1 = -h * self.dq[1] ** 2 - 2.0 * h * self.dq[0] * self.dq[1]
        c2 = h * self.dq[0] ** 2
        return np.array([c1, c2])

    def step(self, torque) -> tuple[np.ndarray, float, float, bool]:
        tau = np.clip(np.asarray(torque, float).reshape(2),
                      -self.max_torque, self.max_torque)
        ddq = np.linalg.solve(self.mass_matrix(),
                              tau - self.coriolis() - self.damping * self.dq)
        self.dq += ddq * self.dt
        self.q += self.dq * self.dt
        # Wrap the angles. Left alone they grow without bound, and a network
        # trained on them sees an input that keeps rising for no physical
        # reason.
        self.q = (self.q + np.pi) % (2.0 * np.pi) - np.pi
        self.t += 1

        dist = float(np.linalg.norm(self.forward() - self.target))
        effort = float(np.sum(tau ** 2)) / (2.0 * self.max_torque ** 2)
        reward = -dist - 0.02 * effort
        done = self.t >= 200
        return self.observe(), reward, dist, done

    def reset(self) -> np.ndarray:
        self.q = self.rng.uniform(-np.pi, np.pi, 2)
        self.dq = np.zeros(2)
        radius = self.max_reach * self.rng.uniform(0.45, 0.95)
        angle = self.rng.uniform(-np.pi, np.pi)
        self.target = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        self.t = 0
        return self.observe()


def pd_reach_teacher(arm: PlanarArm, kp: float = 40.0, kd: float = 6.0) -> np.ndarray:
    """A plain PD controller holding the current joint angles.

    It does not solve the reaching task — it has no idea where the target is —
    so it is not a teacher in the sense the pendulum had one. It is only a
    sanity check that the dynamics integrate and that a reasonable torque
    keeps the arm from flying apart.
    """
    return np.clip(-kp * arm.q - kd * arm.dq, -arm.max_torque, arm.max_torque)


class ArmController:
    """Rank-coded 10-number observation -> spiking pool -> torque pair.

    Ten channels, each with its own pair of disjoint populations, so a channel
    is encoded by *which* cells fire rather than by how hard one group is
    driven. With 10 channels and 4000 neurons that is 100 cells per channel
    per sign, which is enough that two nearby observations do not produce the
    same population.
    """

    def __init__(self, n_neurons: int = 4000, fan_out: int = 32,
                 n_bins: int = 256, n_obs: int = 10,
                 steps_per_action: int = 10, drive: float = 0.9,
                 seed: int = 0, **reservoir_kwargs) -> None:
        self.pool = SparseReservoir(n_neurons, fan_out, seed=seed,
                                    **reservoir_kwargs)
        self.n_neurons = int(n_neurons)
        self.n_bins = int(n_bins)
        self.n_obs = int(n_obs)
        self.steps_per_action = int(steps_per_action)
        self.drive = float(drive)

        rng = np.random.default_rng(seed + 11)
        per_channel = max(1, self.n_neurons // (2 * self.n_obs))
        self.channels = []
        for c in range(self.n_obs):
            pos = np.arange(c * 2 * per_channel, (c * 2 + 1) * per_channel)
            neg = np.arange((c * 2 + 1) * per_channel, (c * 2 + 2) * per_channel)
            thr = rng.uniform(0.0, 1.0, per_channel)
            self.channels.append((pos, neg, thr))
        self.W = np.zeros((2, self.n_bins), np.float32)
        self.last_spikes = np.empty(0, np.int64)
        self.last_rate = 0.0

    def _drive(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx, val = [], []
        for i, (pos, neg, thr) in enumerate(self.channels):
            v = float(np.clip(obs[i], -1.0, 1.0))
            cells, t = (pos, thr) if v > 0 else (neg, thr)
            hit = t < abs(v)
            if hit.any():
                idx.append(cells[hit])
                val.append(np.full(int(hit.sum()), self.drive, np.float32))
        if not idx:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        return np.concatenate(idx), np.concatenate(val)

    def features(self, obs: np.ndarray) -> np.ndarray:
        acc = np.zeros(self.n_bins, np.float32)
        n = 0
        last = np.empty(0, np.int64)
        for _ in range(self.steps_per_action):
            ext, val = self._drive(obs)
            idx = self.pool.step(ext, val) if ext.size else self.pool.step(None)
            last = idx
            if idx.size:
                acc += self.pool.binned_state(idx, self.n_bins)
                n += 1
        # Spikes from the final millisecond only. Keeping the union over all
        # steps made a third of the pool look active at once, which is true
        # over six milliseconds and misleading in a single frame.
        self.last_spikes = last
        self.last_rate = float(last.size) / max(1, self.n_neurons)
        if n:
            acc /= float(n)
        return (acc / (float(np.linalg.norm(acc)) + 1e-8)).astype(np.float32)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.W @ self.features(obs)
