"""A small world and a body, so the brain has something to act on.

The brain on its own can only perceive. Learning needs a consequence, and a
consequence needs an action. This module supplies the smallest honest version
of that loop:

    sense -> act -> the world changes -> reward or nothing

It is deliberately not a game. There is no score, no map, no hint. The body
has eight chemical sensors pointing in different directions, an energy level,
and three actions: turn left, go straight, turn right. Food is scattered; the
only way to find it is to follow the gradient, and the only way to learn that
is to notice which actions are followed by eating.

Nothing here tells the brain what to do. The body reports what it smells and
what happened; the brain decides.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


class ForagingWorld:
    """A continuous 2-D field with scattered food.

    Parameters
    ----------
    n_food:
        How many food items exist at once. Eating one respawns it elsewhere.
    size:
        Side length of the square field.
    sense_radius:
        How far the chemical sensors reach.
    speed, turn:
        Distance per step and radians per turn.
    energy_cost:
        Energy spent per step; eating restores it.
    """

    N_SENSORS = 11          # 8 chemical + distance + energy + heading

    def __init__(self, n_food: int = 12, size: float = 1.0,
                 sense_radius: float = 0.35, speed: float = 0.015,
                 turn: float = 0.35, energy_cost: float = 0.002,
                 seed: int = 0) -> None:
        self.n_food = int(n_food)
        self.size = float(size)
        self.sense_radius = float(sense_radius)
        self.speed = float(speed)
        self.turn = float(turn)
        self.energy_cost = float(energy_cost)
        self.rng = np.random.default_rng(seed)

        self.food = self.rng.random((self.n_food, 2)) * self.size
        self.pos = self.rng.random(2) * self.size
        self.heading = float(self.rng.random() * 2 * math.pi)
        self.energy = 1.0
        self.eaten = 0
        self.total_eaten = 0      # survives death: a lifetime counter
        self.deaths = 0
        self.steps = 0
        self.alive = True

    # -- perception ------------------------------------------------------

    def sense(self) -> np.ndarray:
        """11 numbers: eight directions of smell, nearest distance, energy, heading."""
        out = np.zeros(self.N_SENSORS, np.float32)
        rel = self.food - self.pos                       # (n_food, 2)
        dist = np.linalg.norm(rel, axis=1)
        for k in range(8):
            angle = self.heading + k * (math.pi / 4)
            direction = np.array([math.cos(angle), math.sin(angle)])
            # chemical concentration falls off with distance and with the
            # angle between the sensor and the food
            cosang = (rel @ direction) / (dist + 1e-9)
            near = np.clip(1.0 - dist / self.sense_radius, 0.0, 1.0)
            out[k] = float(np.max(np.clip(cosang, 0, 1) * near))
        out[8] = float(np.clip(dist.min() / self.size, 0.0, 1.0))
        out[9] = float(self.energy)
        out[10] = float((self.heading % (2 * math.pi)) / (2 * math.pi))
        return out

    # -- action ----------------------------------------------------------

    def act(self, action: int) -> float:
        """Apply an action; return the reward (1.0 if food was eaten)."""
        if action == 0:
            self.heading -= self.turn
        elif action == 2:
            self.heading += self.turn

        self.pos = self.pos + self.speed * np.array(
            [math.cos(self.heading), math.sin(self.heading)])
        # walls: bounce back, which is what a real body would feel
        for i in range(2):
            if self.pos[i] < 0.0:
                self.pos[i] = -self.pos[i]
                self.heading += math.pi / 2
            elif self.pos[i] > self.size:
                self.pos[i] = 2 * self.size - self.pos[i]
                self.heading += math.pi / 2

        self.steps += 1
        self.energy -= self.energy_cost

        reward = 0.0
        dist = np.linalg.norm(self.food - self.pos, axis=1)
        hit = int(np.argmin(dist))
        if dist[hit] < 0.04:
            reward = 1.0
            self.eaten += 1
            self.total_eaten += 1
            self.energy = min(1.0, self.energy + 0.35)
            self.food[hit] = self.rng.random(2) * self.size

        if self.energy <= 0.0:
            self.alive = False
        return reward

    def reset(self) -> None:
        """Start a new life. The lifetime total is kept."""
        if not self.alive:
            self.deaths += 1
        self.pos = self.rng.random(2) * self.size
        self.heading = float(self.rng.random() * 2 * math.pi)
        self.energy = 1.0
        self.eaten = 0
        self.steps = 0
        self.alive = True

    def summary(self) -> str:
        return (f"ForagingWorld(food={self.n_food}, eaten={self.eaten}, "
                f"steps={self.steps}, energy={self.energy:.2f})")
