"""A Chrome-dino clone small enough to train in minutes and render to video.

Why this game
-------------
The arm exposed a problem the pendulum hid: reaching has many solutions and
the torque-to-position map is nonlinear, so a linear readout has to approximate
an inverse-kinematics solver. The dino game is the opposite — one binary action,
one hazard, and a state that says exactly how far away it is. It is the
cleanest possible test of whether the whole stack (spiking pool + trained
readout) can hold a closed loop for thousands of frames.

It is also the first task here where success is *watchable*. Bits per character
and mean final distance both need a paragraph of explanation; a dinosaur that
keeps running does not.

Physics
-------
60 frames per second, gravity pulls the dino down, a jump impulse launches it,
and cacti scroll in from the right at a speed that rises with the score. The
dino is a box; a cactus is a box; the run ends when the boxes overlap.

The observation is deliberately generous: the next two obstacles are given with
their distance, height and width. This is not cheating — it is what the game
screen shows a human — and it keeps the learning problem about *timing* rather
than about perception.
"""

from __future__ import annotations

import numpy as np

FPS = 60
# A jump lasts ~24 frames and covers ~120 px at the starting speed. The
# first version used a 32-frame arc, which meant a dino that jumped one
# cactus was still in the air when the next one arrived — the heuristic
# died around frame 500 no matter how its lead was tuned.
GRAVITY = -1.2
JUMP_V = 14.5
SCREEN_W = 600
SCREEN_H = 150
GROUND_Y = 22          # pixels from the bottom
DINO_X = 50
DINO_W = 26
DINO_H = 32


class DinoGame:
    """One dino, cacti, and a score."""

    def __init__(self, seed: int = 0, max_frames: int = 3000,
                 max_speed: float = 15.0) -> None:
        self.rng = np.random.default_rng(seed)
        self.max_frames = int(max_frames)
        # Ceiling on the scroll speed. 15 is the shipped setting; raising it
        # shrinks the reaction window and turns an easy task into one where
        # a controller trained at 15 falls apart — useful for asking whether
        # anything actually improves after the readout is already saturated.
        self.max_speed = float(max_speed)
        # Frames per unit of speed increase. 600 is the shipped ramp, which
        # reaches speed 10 by frame 3000 and never touches the ceiling. Smaller
        # values make the game genuinely harder.
        self.speed_ramp = 600.0
        self.reset()

    # -- lifecycle -------------------------------------------------------

    def reset(self) -> np.ndarray:
        self.y = 0.0            # height above the ground line
        self.vy = 0.0
        self.obstacles: list[list[float]] = []   # [x, w, h]
        self.speed = 5.0
        self.frame = 0
        self.score = 0
        self.dead = False
        self.next_spawn = 60.0
        return self.observe()

    # -- perception ------------------------------------------------------

    def _upcoming(self, limit: int = 2) -> list[list[float]]:
        ahead = [o for o in self.obstacles if o[0] + o[1] > DINO_X - DINO_W]
        ahead.sort(key=lambda o: o[0])
        return ahead[:limit]

    def observe(self) -> np.ndarray:
        """10 numbers: pose, time-to-impact, and the next two cacti.

        Time-to-impact (frames until contact) is given instead of raw speed
        and distance. The jumping rule is "impact in fewer than ~9 frames",
        which in raw units is a *ratio* of distance to speed — a linear readout
        cannot represent it. Measured: decoding the jump label from raw
        (distance, speed) scores R^2 0.05, and behaviour cloning on the pool's
        state scored a 132-frame run against the heuristic's 3000. Dividing
        the two is not a trick: a human reading the screen sees closing speed
        too, and any controller needs that quantity to exist in some form.
        """
        ahead = self._upcoming(2)
        feats = [self.y / 100.0, self.vy / JUMP_V]
        for i in range(2):
            if i < len(ahead):
                x, w, h = ahead[i]
                dist = max(0.0, x - DINO_X)
                tti = dist / max(1e-3, self.speed)
                feats += [min(tti / 60.0, 1.0), dist / SCREEN_W, h / 60.0, w / 30.0]
            else:
                feats += [1.0, 1.0, 0.0, 0.0]
        feats.append(1.0 if self.y > 0.5 else 0.0)
        return np.array(feats[:11] if False else feats, np.float32)

    # -- dynamics --------------------------------------------------------

    def step(self, jump: bool):
        """Advance one frame. Returns (observation, reward, done)."""
        if self.dead:
            return self.observe(), 0.0, True

        # jump only from the ground
        if jump and self.y <= 0.0:
            self.vy = JUMP_V
            self.y = 0.01

        self.vy += GRAVITY
        self.y += self.vy
        if self.y <= 0.0:
            self.y = 0.0
            self.vy = 0.0

        for o in self.obstacles:
            o[0] -= self.speed

        self.obstacles = [o for o in self.obstacles if o[0] + o[1] > -10]

        self.next_spawn -= self.speed
        if self.next_spawn <= 0.0:
            h = float(self.rng.uniform(30.0, 48.0))
            w = float(self.rng.choice([18.0, 22.0, 30.0]))
            self.obstacles.append([SCREEN_W + 10.0, w, h])
            # Gap shrinks as speed rises, but never below a jump's air time.
            # Spacing in *frames*, not pixels. A fixed pixel gap shrinks in
            # time as the scroll speeds up, so at high speed the next cactus
            # arrives while the dino is still airborne and nothing can survive.
            gap_frames = float(self.rng.uniform(34.0, 72.0))
            self.next_spawn = gap_frames * self.speed

        hit = False
        for x, w, h in self.obstacles:
            if (DINO_X + DINO_W > x and DINO_X < x + w
                    and self.y < h and self.y + DINO_H > 0.0):
                hit = True
                break

        self.frame += 1
        self.score += 1
        self.speed = min(self.max_speed, 5.0 + self.frame / self.speed_ramp)

        if hit:
            self.dead = True
            return self.observe(), -1.0, True
        if self.frame >= self.max_frames:
            return self.observe(), 0.1, True
        return self.observe(), 0.1, False

    # -- rendering -------------------------------------------------------

    def render(self) -> np.ndarray:
        """RGB frame, (SCREEN_H, SCREEN_W, 3), uint8.

        Drawn with numpy only: the ground, the dino as a body plus a head, and
        the cacti as bodies plus arms. No fonts, no canvas library, so a frame
        costs a few hundred microseconds and a whole episode renders faster
        than it plays.
        """
        img = np.full((SCREEN_H, SCREEN_W, 3), 247, np.uint8)
        gy = SCREEN_H - GROUND_Y
        img[gy:gy + 2, :] = (83, 83, 83)

        def box(x, y_bottom, w, h, color):
            x0 = int(max(0, x))
            x1 = int(min(SCREEN_W, x + w))
            y1 = int(max(0, SCREEN_H - GROUND_Y - y_bottom))
            y0 = int(max(0, y1 - h))
            if x1 > x0 and y1 > y0:
                img[y0:y1, x0:x1] = color

        dark = (58, 58, 58)
        # dino: body, head, eye, two legs
        box(DINO_X, self.y, DINO_W, DINO_H * 0.72, dark)
        box(DINO_X + DINO_W - 12, self.y + DINO_H * 0.62, 16, 14, dark)
        box(DINO_X + DINO_W - 3, self.y + DINO_H * 0.62 + 8, 3, 3, (247, 247, 247))
        leg = (self.frame // 6) % 2
        box(DINO_X + 3, 0, 5, 6 + (2 if leg else 0), dark)
        box(DINO_X + 13, 0, 5, 8 - (2 if leg else 0), dark)

        for x, w, h in self.obstacles:
            box(x, 0, w, h, (72, 100, 72))
            box(x - 4, 0, 5, h * 0.55, (72, 100, 72))
            box(x + w - 1, 0, 5, h * 0.45, (72, 100, 72))
        return img


def heuristic_jump(obs: np.ndarray, lead: float = 1.0) -> bool:
    """Jump when the first obstacle is close enough that a jump still clears it.

    ``obs`` layout is [y, vy, speed, airborne, dist1, h1, w1, dist2, h2, w2].
    The threshold scales with speed because the dino covers more ground per
    frame at higher speed, so it has to leave earlier.
    """
    if obs[-1] > 0.5:
        return False
    tti = obs[2] * 60.0     # frames until the first cactus arrives
    return tti < 9.0 * lead
