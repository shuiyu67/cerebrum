"""Closing the loop: a brain with a body, learning from consequences.

Until now the brain could only perceive. This adds the missing half — an action
whose outcome feeds back. The learning rule is three-factor, the same shape as
the rest of the engine:

    Δw = learning_rate × (reward − expectation) × presynaptic activity

The middle term is a *global* signal, not a per-synapse error: it says "that
worked" or "that did not", and only the action that was actually taken gets
blamed or credited. No gradient flows back through the spiking network.

What is deliberately left out: any hint about the world. The policy does not
know there is food, does not know the sensors are directional, does not get a
map. It gets a spike pattern and a number.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class Policy:
    """Action selection by population vote — no weight matrix, no argmax over a dot product.

    The brain already partitions its neurons into one group per action. The
    action taken is whichever group *fired more*, counted over the decision
    window. That is the whole mechanism: a race between three populations of
    real neurons. There is no ``W @ rate``, no classifier, no gradient.

    Learning does not happen here either. It happens at the synapses, through
    :class:`~cerebrum.plasticity.Eligibility`: co-active synapses get tagged
    while the animal acts, and dopamine cashes the tags in when food arrives.
    """

    def __init__(self, n_actions: int = 3, explore: float = 0.15,
                 seed: int = 0) -> None:
        self.n_actions = int(n_actions)
        self.explore = float(explore)
        self.rng = np.random.default_rng(seed)
        self.last_action: Optional[int] = None

    def choose(self, counts: np.ndarray) -> int:
        """Pick the action whose neuron group fired most."""
        if self.explore > 0 and self.rng.random() < self.explore:
            action = int(self.rng.integers(self.n_actions))
        else:
            action = int(np.argmax(counts))
        self.last_action = action
        return action


class Agent:
    """A brain in a body, in a world.

    Parameters
    ----------
    brain:
        Any :class:`~cerebrum.core.Brain` whose ``n_inputs`` matches the
        world's sensor count.
    world:
        A :class:`~cerebrum.world.ForagingWorld`.
    steps_per_decision:
        How many milliseconds of brain activity per action. Too short and the
        rate estimate is noisy; too long and the animal is sluggish.
    """

    def __init__(self, brain, world, steps_per_decision: int = 8,
                 explore: float = 0.15, seed: int = 0) -> None:
        if brain.n_outputs != 3:
            raise ValueError("the body has three actions; build the brain with outputs=3")
        self.brain = brain
        self.world = world
        self.steps = int(steps_per_decision)
        self.policy = Policy(3, explore=explore, seed=seed)

    def perceive(self) -> np.ndarray:
        """Spike counts per action group over the decision window."""
        sensor = self.world.sense()
        counts = np.zeros(3, np.float64)
        for _ in range(self.steps):
            action_dist, n = self.brain.step(sensor.tolist())
            counts += np.asarray(action_dist) * max(1, n)
        self.brain.reset_state()
        return counts / float(self.steps)

    def step(self, learn: bool = True):
        counts = self.perceive()
        action = self.policy.choose(counts)
        reward = self.world.act(action)
        surprise = 0.0
        if learn and self.brain.rule("dopamine") is not None:
            # dopamine rises only when something good actually happened; the
            # tagged synapses from the last few hundred milliseconds are the
            # ones that change. No error is propagated anywhere.
            surprise = self.brain.reinforce(reward)
        return action, reward, surprise

    def episode(self, steps: int = 1000, learn: bool = True) -> dict:
        """Run for a while; report what happened.

        A dead body is replaced rather than left to rot: an animal that starves
        does not keep learning on a corpse, and a harness that forgets to
        reset it will report a policy that "learned to die".
        """
        start_eaten = self.world.total_eaten
        start_steps = self.world.steps
        rewards = 0.0
        for _ in range(steps):
            if not self.world.alive:
                self.world.reset()
            _, r, _ = self.step(learn=learn)
            rewards += r
        return {
            "eaten": self.world.total_eaten - start_eaten,
            "steps": self.world.steps - start_steps,
            "reward": rewards,
            "energy": self.world.energy,
            "alive": self.world.alive,
        }
