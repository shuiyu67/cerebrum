"""Render a side-by-side video of the pool playing the dino game.

    python -m examples.dino_render --seed 4242 --out data/dino.mp4

Left half is the spiking pool with its trained readout; right half is the
hand-written heuristic, running the identical obstacle sequence. The two are
driven by the same seed, so every cactus arrives at the same frame in both —
the only difference is the controller.

The video keeps recording for a second after the pool dies, so the moment of
failure is visible rather than the clip ending on it.
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


def load_policy(path: str = "data/dino_pool.npz", threshold: float = 10.0 / 60.0):
    d = np.load(path)
    ctrl = ArmController(n_neurons=int(d["n_neurons"]), fan_out=32,
                         n_bins=int(d["n_bins"]), n_obs=int(d["n_obs"]),
                         steps_per_action=int(d["steps_per_action"]),
                         w_scale=4.0, ginh=20.0, target_frac=0.0, seed=0)
    return ctrl, d["W_mu"], float(d["bias"]), threshold


def render(seed: int = 4242, out: str = "data/dino.mp4", max_frames: int = 3000):
    ctrl, Wmu, bias, th = load_policy()
    pool_game = DinoGame(seed=seed, max_frames=max_frames)
    ref_game = DinoGame(seed=seed, max_frames=max_frames)
    obs_p = pool_game.reset()
    obs_r = ref_game.reset()

    w, h = 600 * SCALE, 150 * SCALE
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (w * 2, h),
         "-r", "60", "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-movflags", "+faststart", out],
        stdin=subprocess.PIPE)

    dead_at = None
    frame = 0
    try:
        for t in range(max_frames):
            if not pool_game.dead:
                jump = ((obs_p[-1] < 0.5)
                        and (float(Wmu @ ctrl.features(obs_p) + bias) < th))
                obs_p, _r, done = pool_game.step(jump)
                if done and dead_at is None:
                    dead_at = t
            else:
                pool_game.step(False)

            if not ref_game.dead:
                obs_r, _r, _done = ref_game.step(heuristic_jump(obs_r, lead=1.3))
            else:
                ref_game.step(False)

            canvas = Image.new("RGB", (w * 2, h), (247, 247, 247))
            canvas.paste(Image.fromarray(pool_game.render()).resize((w, h), Image.NEAREST), (0, 0))
            canvas.paste(Image.fromarray(ref_game.render()).resize((w, h), Image.NEAREST), (w, 0))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle([0, 0, w * 2 - 1, 22], fill=(255, 255, 255))
            draw.text((8, 5), "POOL (spiking reservoir)  %d" % pool_game.score,
                      fill=(180, 30, 30))
            draw.text((w + 8, 5), "HEURISTIC reference  %d" % ref_game.score,
                      fill=(30, 60, 180))
            draw.line([(w, 0), (w, h)], fill=(200, 200, 200), width=2)
            if pool_game.dead:
                draw.text((8, h - 18), "FELL OVER", fill=(180, 30, 30))
            proc.stdin.write(canvas.tobytes())
            frame += 1
            if dead_at is not None and t > dead_at + 540:
                break
    finally:
        proc.stdin.close()
        proc.wait()

    print("wrote %s  %d frames (%.1f s at 60fps)" % (out, frame, frame / 60.0))
    print("pool score %d, heuristic score %d"
          % (pool_game.score, ref_game.score))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default="data/dino.mp4")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    render(seed=args.seed, out=args.out)
