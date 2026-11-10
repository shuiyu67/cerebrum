"""Proximal policy optimisation, hand-written in numpy.

Why hand-written
----------------
The published reservoir controllers use PPO out of a library, which is fine
until the target is a microcontroller. This one is 150 lines of numpy with no
autograd, no graph, no dependency — the same reason the pool itself is numpy.
The gradients are written out by hand, which is possible here only because the
policy and the value function are both *linear* in the features.

What it learns
--------------
A Gaussian policy over continuous actions. The mean is a linear map of the
state features, the standard deviation is a learned constant per action
dimension. Training maximises the clipped surrogate objective, and the value
function is fitted with plain least squares by gradient descent.

Nothing here touches the pool. This trains the readout — which is exactly the
part the published work trains and the part the frozen-pool experiments left
unexplored.
"""

from __future__ import annotations

import numpy as np


class NumpyPPO:
    """PPO with a linear Gaussian policy and a linear critic.

    Parameters follow the usual defaults; the ones that matter for a task this
    small are ``lr`` (too high and the policy collapses to a single action in
    a few updates) and ``ent_coef`` (too low and it stops exploring before it
    has found the target).
    """

    def __init__(self, n_features: int, n_actions: int, *, lr: float = 3e-3,
                 gamma: float = 0.99, lam: float = 0.95, clip: float = 0.2,
                 epochs: int = 8, vf_coef: float = 0.5,
                 ent_coef: float = 0.01, max_grad: float = 1.0,
                 seed: int = 0) -> None:
        self.n_features = int(n_features)
        self.n_actions = int(n_actions)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.clip = float(clip)
        self.epochs = int(epochs)
        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)
        self.max_grad = float(max_grad)
        self.rng = np.random.default_rng(seed)

        self.W_mu = np.zeros((self.n_actions, self.n_features), np.float32)
        self.log_std = np.full(self.n_actions, -1.0, np.float32)
        self.W_v = np.zeros(self.n_features, np.float32)
        self.updates = 0

    # -- acting ----------------------------------------------------------

    def distribution(self, features: np.ndarray):
        mu = self.W_mu @ features
        std = np.exp(self.log_std)
        return mu, std

    def act(self, features: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """Sample an action; return it with its log-probability and the mean.

        The mean is returned so the caller can evaluate a deterministic policy
        at test time without sampling noise.
        """
        mu, std = self.distribution(features)
        z = self.rng.standard_normal(self.n_actions)
        action = mu + std * z
        logp = float(np.sum(-0.5 * z ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)))
        return action.astype(np.float32), logp, mu

    def value(self, features: np.ndarray) -> float:
        return float(self.W_v @ features)

    # -- learning --------------------------------------------------------

    @staticmethod
    def _gae(rewards, values, dones, gamma: float, lam: float):
        """Generalised advantage estimation.

        One step of the usual backward pass. ``values`` has one extra entry:
        the bootstrap value of the state after the last transition.
        """
        T = len(rewards)
        adv = np.zeros(T, np.float32)
        last = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - float(dones[t])
            next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value * nonterminal - values[t]
            last = delta + gamma * lam * nonterminal * last
            adv[t] = last
        returns = adv + np.asarray(values[:-1], np.float32)
        return adv, returns

    def update(self, features, actions, logp_old, rewards, dones) -> dict:
        """One PPO round over a finished batch of transitions."""
        features = np.asarray(features, np.float32)
        actions = np.asarray(actions, np.float32)
        logp_old = np.asarray(logp_old, np.float32)
        rewards = np.asarray(rewards, np.float32)
        dones = np.asarray(dones, np.float32)

        values = np.array([self.value(f) for f in features]
                          + [self.value(features[-1])], np.float32)
        adv, returns = self._gae(rewards, values, dones, self.gamma, self.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        std = np.exp(self.log_std)
        loss_total = 0.0
        for _ in range(self.epochs):
            g_mu = np.zeros_like(self.W_mu)
            g_logstd = np.zeros_like(self.log_std)
            g_v = np.zeros_like(self.W_v)
            loss_epoch = 0.0

            for i in range(len(features)):
                f = features[i]
                mu = self.W_mu @ f
                z = (actions[i] - mu) / std
                logp = float(np.sum(-0.5 * z ** 2 - np.log(std)
                                    - 0.5 * np.log(2 * np.pi)))
                ratio = float(np.exp(logp - logp_old[i]))
                a = float(adv[i])

                # Clipped surrogate: the gradient only flows when the ratio is
                # inside the trust region, or when moving further would make
                # the objective worse anyway.
                unclipped = ratio * a
                clipped = float(np.clip(ratio, 1.0 - self.clip,
                                        1.0 + self.clip)) * a
                if unclipped <= clipped:
                    dlogp = -a * ratio
                    loss_epoch -= min(unclipped, clipped)
                else:
                    dlogp = 0.0
                    loss_epoch -= min(unclipped, clipped)

                # d log pi / d mu = z / std ; d log pi / d log_std = z^2 - 1
                dmu = dlogp * z / std
                g_mu += np.outer(dmu, f)
                g_logstd += dlogp * (z ** 2 - 1.0)

                # Value regression, plain squared error.
                v = float(self.W_v @ f)
                dv = v - float(returns[i])
                g_v += dv * f
                loss_epoch += 0.5 * self.vf_coef * dv ** 2

                # Entropy bonus keeps the policy from collapsing to a point.
                g_logstd += self.ent_coef * np.ones(self.n_actions)

            n = float(len(features))
            for g in (g_mu, g_logstd, g_v):
                g /= n
            self._clip_step(g_mu, 'W_mu')
            self._clip_step(g_logstd, 'log_std')
            self._clip_step(self.vf_coef * g_v, 'W_v')
            loss_total = loss_epoch / n

        self.updates += 1
        return {"loss": float(loss_total),
                "entropy": float(np.sum(self.log_std + 0.5 * np.log(2 * np.pi * np.e))),
                "std": float(np.exp(self.log_std).mean())}

    def _clip_step(self, grad: np.ndarray, name: str) -> None:
        norm = float(np.linalg.norm(grad))
        if norm > self.max_grad:
            grad = grad * (self.max_grad / (norm + 1e-12))
        setattr(self, name, getattr(self, name) - self.lr * grad)
