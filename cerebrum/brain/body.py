"""A body to have a brain in.

A brain with no body can only be measured on how well it reconstructs its own
input. That is a closed loop with nothing at stake, and it cannot distinguish
a system that understands from one that has memorised a sequence.

This is the smallest environment that puts something at stake: a grid with
food that relieves hunger, and a predator that hurts. Six actions, sixteen
senses, two outcomes that matter. Everything the brain does has to be judged
against whether the animal eats and survives, not against reconstruction
error.

The sensors are deliberately low-dimensional and pre-processed — bearings and
proximities rather than pixels. The point of the exercise is the architecture
above the senses, and a retina would only add a second research programme.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

ACTIONS = ("forward", "turn_left", "turn_right", "eat", "flee", "rest")
N_ACTIONS = len(ACTIONS)

# Sensor layout (16 channels, matching the default thalamic channel count)
# 0-3  food bearing, four quadrants, intensity falls with distance
# 4-7  predator bearing, four quadrants
# 8    hunger
# 9    fatigue
# 10   energy
# 11   fear
# 12   food directly ahead
# 13   wall ahead
# 14   predator adjacent
# 15   time since last meal (saturating)
N_SENSE = 16


class Body:
    """Grid world with food, a predator, and interoception."""

    def __init__(self, size: int = 16, n_food: int = 5, seed: int = 0,
                 predator: bool = True) -> None:
        self.size = int(size)
        self.predator_on = bool(predator)
        self.rng = np.random.default_rng(seed)
        self.reset()

    # -- lifecycle -------------------------------------------------------

    def reset(self) -> np.ndarray:
        self.x = self.size // 2
        self.y = self.size // 2
        self.heading = 0            # 0=E, 1=NE, 2=N, ... clockwise
        self.food: list = []
        self.food = [self._free_cell() for _ in range(5)]
        self.px = int(self.rng.integers(0, self.size))
        self.py = int(self.rng.integers(0, self.size))
        self.energy = 1.0
        self.hunger = 0.3
        self.fatigue = 0.1
        self.fear = 0.0
        self.since_meal = 0
        self.eaten = 0
        self.caught = 0
        self.steps = 0
        self.alive = True
        return self.sense()

    def _free_cell(self) -> tuple:
        for _ in range(200):
            p = (int(self.rng.integers(0, self.size)),
                 int(self.rng.integers(0, self.size)))
            if p != (self.x, self.y) and p not in self.food:
                return p
        return (0, 0)

    # -- geometry --------------------------------------------------------

    def _rel(self, tx: float, ty: float):
        """Bearing and distance from the agent to a target."""
        dx = tx - self.x
        dy = ty - self.y
        dist = float(np.hypot(dx, dy))
        ang = np.arctan2(dy, dx)
        # quadrant relative to heading (8 directions -> 4 quadrants)
        rel = (ang - self.heading * np.pi / 4) % (2 * np.pi)
        q = int(((rel + np.pi / 4) % (2 * np.pi)) // (np.pi / 2)) % 4
        return q, dist, ang

    def sense(self) -> np.ndarray:
        s = np.zeros(N_SENSE, np.float32)
        for fx, fy in self.food:
            q, d, _ = self._rel(fx, fy)
            s[q] = max(s[q], 1.0 / (1.0 + d))
        if self.predator_on:
            q, d, _ = self._rel(self.px, self.py)
            s[4 + q] = max(s[4 + q], 1.0 / (1.0 + d))
            s[14] = 1.0 if d < 1.6 else 0.0
        s[8] = self.hunger
        s[9] = self.fatigue
        s[10] = self.energy
        s[11] = self.fear
        # look one cell ahead
        ax = int(round(self.x + np.cos(self.heading * np.pi / 4)))
        ay = int(round(self.y + np.sin(self.heading * np.pi / 4)))
        s[12] = 1.0 if (ax, ay) in self.food else 0.0
        s[13] = 1.0 if not (0 <= ax < self.size and 0 <= ay < self.size) else 0.0
        s[15] = min(1.0, self.since_meal / 200.0)
        return s

    # -- interaction -----------------------------------------------------

    def act(self, action: int) -> tuple:
        """Apply an action; returns ``(sensors, reward, done, info)``."""
        self.steps += 1
        self.since_meal += 1
        reward = -0.002                      # a small metabolic cost per step
        info: dict = {"ate": False, "hurt": False}

        if action == 0:                      # forward
            dx = int(round(np.cos(self.heading * np.pi / 4)))
            dy = int(round(np.sin(self.heading * np.pi / 4)))
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                self.x, self.y = nx, ny
            self.energy -= 0.004
            self.fatigue = min(1.5, self.fatigue + 0.003)
        elif action == 1:
            self.heading = (self.heading + 1) % 8
        elif action == 2:
            self.heading = (self.heading - 1) % 8
        elif action == 3:                    # eat
            if (self.x, self.y) in self.food or self._adjacent_food():
                target = ((self.x, self.y) if (self.x, self.y) in self.food
                          else self._adjacent_food())
                self.food.remove(target)
                self.food.append(self._free_cell())
                self.energy = min(1.5, self.energy + 0.5)
                self.hunger = max(0.0, self.hunger - 0.45)
                self.since_meal = 0
                self.eaten += 1
                reward = 1.0
                info["ate"] = True
        elif action == 4:                    # flee: two cells backwards
            dx = -int(round(np.cos(self.heading * np.pi / 4)))
            dy = -int(round(np.sin(self.heading * np.pi / 4)))
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                self.x, self.y = nx, ny
            self.energy -= 0.008
            self.fatigue = min(1.5, self.fatigue + 0.006)
        elif action == 5:                    # rest
            self.energy = min(1.2, self.energy + 0.003)
            self.fatigue = max(0.0, self.fatigue - 0.004)

        # hunger and energy drift
        self.energy = float(np.clip(self.energy - 0.0012, 0.0, 1.5))
        self.hunger = float(np.clip(self.hunger + 0.0006, 0.0, 1.5))

        if self.predator_on:
            if self.steps % 5 == 0:
                if self.px < self.x:
                    self.px += 1
                elif self.px > self.x:
                    self.px -= 1
                if self.py < self.y:
                    self.py += 1
                elif self.py > self.y:
                    self.py -= 1
            if abs(self.px - self.x) <= 1 and abs(self.py - self.y) <= 1:
                reward = -0.6
                self.fear = 1.0
                self.caught += 1
                info["hurt"] = True
                # the animal is not killed, it is moved away — death ends
                # the experiment, and one trial per lifetime teaches nothing
                self._escape()
            else:
                self.fear = max(0.0, self.fear - 0.01)
        else:
            self.fear = max(0.0, self.fear - 0.01)

        if self.energy <= 0.0:
            self.alive = False
            reward -= 1.0
            info["starved"] = True

        return self.sense(), float(reward), not self.alive, info

    def _adjacent_food(self):
        for f in self.food:
            if abs(f[0] - self.x) <= 1 and abs(f[1] - self.y) <= 1:
                return f
        return None

    def _escape(self) -> None:
        self.x = int(self.rng.integers(0, self.size))
        self.y = int(self.rng.integers(0, self.size))
        self.px = int(self.rng.integers(0, self.size))
        self.py = int(self.rng.integers(0, self.size))

    def summary(self) -> str:
        return (f"Body(pos {self.x},{self.y}, energy {self.energy:.2f}, "
                f"hunger {self.hunger:.2f}, eaten {self.eaten}, "
                f"caught {self.caught})")
