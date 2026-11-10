"""Render the dino game with the pool's own activity underneath it.

    python -m examples.dino_render --seed 4259 --out data/dino.mp4

Top: the game, driven by the trained readout. Bottom: the firing pattern of the
spiking pool that produced each decision, laid out as a 120 x 50 sheet, with a
short afterglow so a spike is visible for a few frames rather than one.

When the dino dies the pool stops being driven, so the sheet fades to black
over about a dozen frames and stays dark. That is not a rendering artifact —
it is what an undriven pool does.
"""

from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from cerebrum.control.arm import ArmController
from cerebrum.control.dino import DinoGame, heuristic_jump

SCALE = 2
GAME_W, GAME_H = 600, 150
BRAIN_H = 200
BRAIN_ROWS = 50


def load_policy(path: str = "data/dino_pool.npz", threshold: float = 13.0 / 60.0,
                smooth: float = 0.4):
    """Load the readout, its decision threshold, and the prediction smoother.

    Smoothing is not cosmetic. The readout estimates time-to-impact with about
    three frames of error, and a single noisy estimate near the threshold fires
    a jump too early or too late. An exponential average over consecutive
    estimates cut the mean score from 192 to 1952 frames at threshold 13 — the
    difference between a controller that usually dies early and one that often
    runs the full 3000.
    """
    d = np.load(path)
    ctrl = ArmController(n_neurons=int(d["n_neurons"]), fan_out=32,
                         n_bins=int(d["n_bins"]), n_obs=int(d["n_obs"]),
                         steps_per_action=int(d["steps_per_action"]),
                         w_scale=4.0, ginh=20.0, target_frac=0.0, seed=0)
    return ctrl, d["W_mu"], float(d["bias"]), threshold, smooth


def brain_image(glow: np.ndarray, rows: int, perm: np.ndarray) -> np.ndarray:
    """Glow array (one value per neuron) -> an (BRAIN_H, GAME_W, 3) sheet.

    Cells are laid out through a fixed random permutation. In index order the
    sheet shows horizontal stripes, because neighbouring indices belong to the
    same input channel and fire together — true, but it reads as scan lines
    rather than as a population.
    """
    cols = glow.size // rows
    g = glow[perm].reshape(rows, cols)
    # A dim floor keeps the sheet from being pure black, so the firing pattern
    # reads as activity on a substrate rather than as light in a void.
    rgb = (g[:, :, None] * np.array([70.0, 205.0, 255.0], np.float32)
           + np.array([10.0, 12.0, 18.0], np.float32)).astype(np.uint8)
    return np.repeat(np.repeat(rgb, GAME_W // cols, axis=1),
                     BRAIN_H // rows, axis=0)


def render(seed: int = 4259, out: str = "data/dino.mp4", max_frames: int = 3000,
           after_death: int = 540) -> None:
    ctrl, Wmu, bias, th, smooth = load_policy()
    ctrl.pool.reset()
    game = DinoGame(seed=seed, max_frames=max_frames)
    obs = game.reset()
    tti_est = 1.0

    n = ctrl.n_neurons
    rows = BRAIN_ROWS
    cols = n // rows
    glow = np.zeros(rows * cols, np.float32)
    perm = np.random.default_rng(7).permutation(rows * cols)

    W, H = GAME_W * SCALE, (GAME_H + BRAIN_H) * SCALE
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
         "-r", "60", "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-movflags", "+faststart", out],
        stdin=subprocess.PIPE)

    dead_at = None
    frames = 0
    peak_rate = 0.0
    try:
        for t in range(max_frames):
            if not game.dead:
                raw = float(Wmu @ ctrl.features(obs) + bias)
                tti_est = smooth * tti_est + (1.0 - smooth) * raw
                obs, _r, done = game.step((obs[-1] < 0.5) and (tti_est < th))
                if done and dead_at is None:
                    dead_at = t
                peak_rate = max(peak_rate, ctrl.last_rate)
            else:
                game.step(False)

            glow *= 0.80
            if ctrl.last_spikes.size and not game.dead:
                glow[ctrl.last_spikes] = 1.0

            top = game.render()
            bottom = brain_image(glow, rows, perm)
            canvas = Image.fromarray(np.vstack([top, bottom]))
            canvas = canvas.resize((W, H), Image.NEAREST)
            draw = ImageDraw.Draw(canvas)
            draw.rectangle([0, 0, W - 1, 26], fill=(255, 255, 255))
            live = 0.0 if game.dead else ctrl.last_rate * 100.0
            draw.text((10, 6), "POOL  score %d   |   %d spiking neurons   |   %.1f%% active"
                      % (game.score, n, live), fill=(60, 60, 60))
            draw.line([(0, GAME_H * SCALE), (W, GAME_H * SCALE)],
                      fill=(200, 200, 200), width=1)
            if game.dead:
                draw.text((10, GAME_H * SCALE + 6), "DINO DOWN — pool undriven, activity decays",
                          fill=(255, 120, 120))
            proc.stdin.write(canvas.tobytes())
            frames += 1
            if dead_at is not None and t > dead_at + after_death:
                break
    finally:
        proc.stdin.close()
        proc.wait()

    print("wrote %s  %d frames (%.1f s)" % (out, frames, frames / 60.0))
    print("pool score %d, peak activity %.1f%%" % (game.score, peak_rate * 100))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4259)
    ap.add_argument("--out", default="data/dino.mp4")
    ap.add_argument("--after-death", type=int, default=540,
                    help="frames to keep recording after the dino dies")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    render(seed=args.seed, out=args.out, after_death=args.after_death)
