"""Closed-loop control: a spiking pool holding an inverted pendulum up.

Why this task and not another language model
--------------------------------------------
The language model measured the pool at 0.05 bits of contribution and lost to a
counting table. That is a statement about character prediction, not about the
pool: bits-per-character is a language-model yardstick, and applying it to a
spiking network is measuring the wrong thing. A reservoir is not built to
predict tokens; it is built to hold a state and act on it, which is what
control needs and what the reservoir literature actually demonstrates.

So the yardstick changes from bits to **time standing up**. A comparison here
is legible without any statistics: the pole either falls over or it does not.

The task
--------
An inverted pendulum starts near upright and is unstable — the gravitational
term ``(g/L) sin(theta)`` pushes it further from vertical the further it
already is. The only control is a torque, clipped at ``max_torque``. A randomly
weighted controller holds it for a few tenths of a second; a proportional-
derivative controller holds it indefinitely. The question is where the spiking
pool lands without ever computing a gradient through its own recurrence.

The readout is still one linear layer, trained by ridge regression on a
teacher's torque. That is deliberate: the point is not to prove that learning
can be avoided, it is to ask whether the pool's fixed dynamics are rich enough
to *support* a controller once the last layer is fitted. If a linear map of the
pool state can hold the pole, the dynamics carry the control signal. If only a
direct linear map of the raw state can hold it, the pool added nothing and the
result says so.
"""

from __future__ import annotations

import numpy as np

from ..lm.reservoir import SparseReservoir


class InvertedPendulum:
    """A torque-driven pole on a cart, angle measured from upright.

    ``theta = 0`` is balanced, positive is leaning right. Semi-implicit Euler
    at ``dt``; the damping term stands in for the cart's friction and keeps a
    purely proportional controller from oscillating forever.
    """

    def __init__(self, g: float = 9.81, length: float = 0.5, mass: float = 1.0,
                 damping: float = 0.5, dt: float = 0.01,
                 max_torque: float = 10.0, noise: float = 0.0,
                 seed: int = 0) -> None:
        self.g = float(g)
        self.L = float(length)
        self.m = float(mass)
        self.damping = float(damping)
        self.dt = float(dt)
        self.max_torque = float(max_torque)
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)
        self.theta = 0.0
        self.omega = 0.0
        self.t = 0
        self.fallen = False

    @property
    def inertia(self) -> float:
        return self.m * self.L ** 2

    def reset(self, theta: float | None = None) -> np.ndarray:
        self.theta = (float(self.rng.uniform(-0.12, 0.12))
                      if theta is None else float(theta))
        self.omega = float(self.rng.uniform(-0.05, 0.05))
        self.t = 0
        self.fallen = False
        return self.observe()

    def observe(self) -> np.ndarray:
        return np.array([self.theta, self.omega], np.float32)

    def step(self, torque: float) -> tuple[np.ndarray, float]:
        u = float(np.clip(torque, -self.max_torque, self.max_torque))
        acc = ((self.g / self.L) * np.sin(self.theta)
               - self.damping * self.omega
               + u / self.inertia)
        self.omega += acc * self.dt
        self.theta += self.omega * self.dt
        if self.noise:
            self.omega += float(self.rng.normal(0.0, self.noise))
        self.t += 1
        # 0.4 rad is 23 degrees. Past that the pole is not coming back with a
        # torque limit of 10; keeping the threshold tight makes the score mean
        # "held it upright" rather than "kept it on the table".
        if abs(self.theta) > 0.4:
            self.fallen = True
        return self.observe(), u


def pd_teacher(obs: np.ndarray, kp: float = 25.0, kd: float = 6.0,
               limit: float = 10.0) -> float:
    """The reference controller. Sign convention: positive theta means leaning
    right, so the torque has to push left."""
    theta, omega = float(obs[0]), float(obs[1])
    return float(np.clip(-kp * theta - kd * omega, -limit, limit))


class PoolController:
    """Rank-coded state -> spiking pool -> linear readout -> torque.

    The state is two scalars, so each is rank-coded onto its own pair of
    disjoint populations: a positive value drives the "positive" population up
    to a threshold set per cell, a negative value drives the "negative" one.
    That is the same coding the memory-capacity measurement uses, and it exists
    for the same reason — driving a whole group at one uniform current makes
    every cell in it fire in lockstep, so the population reports its refractory
    phase rather than the value it was given.
    """

    def __init__(self, n_neurons: int = 4000, fan_out: int = 32,
                 n_bins: int = 256, steps_per_action: int = 10,
                 hold_ms: int = 1, drive: float = 0.9,
                 theta_scale: float = 0.5, omega_scale: float = 2.0,
                 use_omega: bool = True, seed: int = 0,
                 **reservoir_kwargs) -> None:
        self.pool = SparseReservoir(n_neurons, fan_out, seed=seed,
                                    **reservoir_kwargs)
        self.n_neurons = int(n_neurons)
        self.n_bins = int(n_bins)
        self.steps_per_action = int(steps_per_action)
        self.hold_ms = int(hold_ms)
        self.drive = float(drive)
        self.theta_scale = float(theta_scale)
        self.omega_scale = float(omega_scale)
        # Partial observability switch. With ``use_omega=False`` the pool sees
        # only the angle, which is the honest test of what a recurrent pool is
        # for: the derivative is not in the input, but it is recoverable from
        # the angle's recent history, and the pool's state is that history.
        # A memoryless linear controller cannot recover it, so the comparison
        # has a clear expected sign before it is run.
        self.use_omega = bool(use_omega)

        rng = np.random.default_rng(seed + 7)
        q = max(1, self.n_neurons // 8)
        self.groups = [(np.arange(0, q), rng.uniform(0, 1, q)),
                       (np.arange(q, 2 * q), rng.uniform(0, 1, q)),
                       (np.arange(2 * q, 3 * q), rng.uniform(0, 1, q)),
                       (np.arange(3 * q, 4 * q), rng.uniform(0, 1, q))]
        self.W = np.zeros(self.n_bins, np.float32)

    def _drive(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = float(np.clip(obs[0] / self.theta_scale, -1.0, 1.0))
        omega = float(np.clip(obs[1] / self.omega_scale, -1.0, 1.0))
        idx, val = [], []
        channels = [(self.groups[0], theta), (self.groups[1], -theta)]
        if self.use_omega:
            channels += [(self.groups[2], omega), (self.groups[3], -omega)]
        for (cells, thr), v in channels:
            if v > 0.0:
                hit = thr < v
                idx.append(cells[hit])
                val.append(np.full(int(hit.sum()), self.drive, np.float32))
        if not idx:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        return np.concatenate(idx), np.concatenate(val)

    def features(self, obs: np.ndarray) -> np.ndarray:
        """Run the pool for one control period and average its projection."""
        acc = np.zeros(self.n_bins, np.float32)
        n = 0
        for _ in range(self.steps_per_action):
            for _ in range(self.hold_ms):
                ext, val = self._drive(obs)
                idx = (self.pool.step(ext, val) if ext.size
                       else self.pool.step(None))
                if idx.size:
                    acc += self.pool.binned_state(idx, self.n_bins)
                    n += 1
        if n:
            acc /= float(n)
        norm = float(np.linalg.norm(acc)) + 1e-8
        return (acc / norm).astype(np.float32)

    def act(self, obs: np.ndarray) -> float:
        return float(self.W @ self.features(obs))
