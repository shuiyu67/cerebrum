"""Snake, as a sensory stream for a spiking brain.

The point of this environment is not the game. It is to give the brain a
closed loop that punishes and rewards, with senses it could plausibly have and
actions it can actually take. Three design choices matter:

* **Relative actions.** The snake turns left, goes straight, or turns right.
  A fly does not decide "north"; it decides "turn towards the smell". Absolute
  directions would put the compass in the environment instead of the brain.
* **A local view.** The brain sees an 11x11 patch around its own head, not the
  whole board. Anything outside that patch is unknown, which is what makes
  memory worth having.
* **Three channels.** Danger (wall + own body), food, and the direction it is
  already heading. The third channel is what lets it tell "food on my left"
  from "food behind me".
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

__all__ = ["Snake"]


class Snake:
    """A grid snake with a local, three-channel view.

    Parameters
    ----------
    size:
        Board is ``size x size`` with walls all around.
    view:
        Odd side length of the local patch. 11 gives a 363-wide sensor vector.
    step_penalty:
        Reward subtracted every step. Without it the snake can circle forever
        and still collect the same reward per food, so there is no pressure to
        be efficient.
    """

    ACTIONS = 3          # 0 turn left, 1 straight, 2 turn right

    def __init__(self, size: int = 20, view: int = 11,
                 step_penalty: float = 0.002, seed: Optional[int] = None,
                 max_steps: Optional[int] = None) -> None:
        self.size = int(size)
        self.view = int(view) | 1        # force odd
        self.half = self.view // 2
        self.step_penalty = float(step_penalty)
        self.rng = np.random.default_rng(seed)
        self.max_steps = int(max_steps or size * size * 4)
        self.n_sensors = self.view * self.view * 3
        self.reset()

    # -- life cycle ------------------------------------------------------

    def reset(self) -> np.ndarray:
        c = self.size // 2
        self.body: List[Tuple[int, int]] = [(c, c), (c - 1, c), (c - 2, c)]
        self.heading = 0                  # 0 up, 1 right, 2 down, 3 left
        self.steps = 0
        self.alive = True
        self.ate = 0
        self._place_food()
        return self.sense()

    def _place_food(self) -> None:
        free = [(x, y) for x in range(self.size) for y in range(self.size)
                if (x, y) not in self.body]
        if not free:
            self.food = None
            return
        self.food = free[int(self.rng.integers(len(free)))]

    # -- perception ------------------------------------------------------

    def sense(self) -> np.ndarray:
        """A ``view x view x 3`` patch, rotated so "up" is where we are going.

        Rotating into the snake's own frame is what makes the same sensor
        pattern mean the same thing after a turn — otherwise the brain would
        have to relearn every orientation separately.
        """
        hx, hy = self.body[0]
        dx, dy = _DIRS[self.heading]
        # right-hand vector, for the lateral axis of the patch
        rx, ry = -dy, dx
        body = set(self.body[1:])
        out = np.zeros((self.view, self.view, 3), np.float32)
        h = self.half
        for j in range(self.view):
            for i in range(self.view):
                fwd = j - h
                lat = i - h
                x = hx + dx * fwd + rx * lat
                y = hy + dy * fwd + ry * lat
                if not (0 <= x < self.size and 0 <= y < self.size):
                    out[j, i, 0] = 1.0            # wall
                    continue
                if (x, y) in body:
                    out[j, i, 0] = 1.0            # self
                if self.food is not None and (x, y) == self.food:
                    out[j, i, 1] = 1.0            # food
                out[j, i, 2] = 1.0 if fwd > 0 else 0.0   # ahead of me
        return out.reshape(-1)

    # -- action ----------------------------------------------------------

    def act(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Apply one of the three relative actions. Returns (sense, reward, done)."""
        if not self.alive:
            return self.sense(), 0.0, True

        if action == 0:
            self.heading = (self.heading - 1) % 4
        elif action == 2:
            self.heading = (self.heading + 1) % 4

        dx, dy = _DIRS[self.heading]
        hx, hy = self.body[0]
        nx, ny = hx + dx, hy + dy
        self.steps += 1

        reward = -self.step_penalty
        if not (0 <= nx < self.size and 0 <= ny < self.size):
            self.alive = False
            return self.sense(), reward - 1.0, True
        if (nx, ny) in self.body[:-1]:
            self.alive = False
            return self.sense(), reward - 1.0, True

        self.body.insert(0, (nx, ny))
        if self.food is not None and (nx, ny) == self.food:
            self.ate += 1
            reward += 1.0
            self._place_food()
        else:
            self.body.pop()

        if self.steps >= self.max_steps:
            self.alive = False
            return self.sense(), reward, True
        return self.sense(), reward, False

    # -- convenience -----------------------------------------------------

    def render(self) -> str:
        rows = []
        body = set(self.body)
        head = self.body[0]
        for y in range(self.size):
            row = []
            for x in range(self.size):
                if (x, y) == head:
                    row.append("@")
                elif (x, y) in body:
                    row.append("o")
                elif self.food is not None and (x, y) == self.food:
                    row.append("*")
                else:
                    row.append(".")
            rows.append("".join(row))
        return "\n".join(rows)

    def state(self) -> dict:
        return {
            "body": list(self.body),
            "food": self.food,
            "heading": self.heading,
            "ate": self.ate,
            "steps": self.steps,
            "alive": self.alive,
            "size": self.size,
        }


_DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))
