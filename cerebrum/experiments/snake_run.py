"""Let a spiking brain play snake, and record everything it does.

Two questions this is built to answer honestly:

* can a brain of fly scale (139k neurons, 50M synapses) be driven by a real
  sensory stream at a speed that makes learning measurable at all;
* does adding spike-timing eligibility plus dopamine actually beat a brain that
  does not learn, or does it only look like it does.

The recorder writes one JSON file per run containing every frame: the snake's
body, the action taken, the reward, and the *indices of every neuron that
fired* that step. Indices rather than a dense bitmap, because 139k neurons a
frame would be 139 KB per step and almost all of them are silent.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import numpy as np

from cerebrum.core import Brain
from cerebrum.plasticity import Dopamine, Eligibility
from cerebrum.snake import Snake


class SnakePlayer:
    """Drives a snake with population votes from a spiking brain.

    Action selection has no classifier in it: the brain's actuators are already
    three groups of neurons, and the group that fires most during the decision
    window is the action. That is the whole mechanism, and it means the only
    thing that can improve behaviour is a change in synaptic weights.
    """

    def __init__(self, brain: Brain, snake: Snake, steps_per_decision: int = 5,
                 explore: float = 0.1, seed: int = 0) -> None:
        self.brain = brain
        self.snake = snake
        self.steps = int(steps_per_decision)
        self.explore = float(explore)
        self.rng = np.random.default_rng(seed)
        self.group = brain.group_size

    def counts(self, sensor: np.ndarray) -> np.ndarray:
        """Spikes per action group over the decision window."""
        acc = np.zeros(3, np.float64)
        for _ in range(self.steps):
            dist, n = self.brain.step(sensor.tolist())
            acc += np.asarray(dist) * max(1, n)
        return acc

    def choose(self, counts: np.ndarray) -> int:
        if self.explore > 0 and self.rng.random() < self.explore:
            return int(self.rng.integers(3))
        return int(np.argmax(counts))

    def step(self, learn: bool = True):
        sensor = self.snake.sense()
        counts = self.counts(sensor)
        action = self.choose(counts)
        _, reward, done = self.snake.act(action)
        surprise = 0.0
        if learn and self.brain.rule("dopamine") is not None:
            surprise = self.brain.reinforce(reward)
        return sensor, counts, action, reward, done, surprise


def state_key(sensor: np.ndarray) -> int:
    """Coarse bucket for a sensory vector, for measuring habit formation.

    Six bands, three levels each — 729 states. Coarse on purpose: the question
    is not whether it can tell two inputs apart, it is whether *the same*
    input stops producing a different action every time.
    """
    # Three bands, three levels: 27 states. The sensor vector is 11x11x3 = 363,
    # so three bands divide evenly (121 each). Finer than this and every state
    # has too few samples for the entropy estimate to mean anything — measured,
    # a 729-state version scored random actions at 1.19 bits against a true
    # 1.585, purely from small-sample bias.
    band = sensor.reshape(3, -1).sum(axis=1)
    lv = np.digitize(band, [0.5, 2.5])
    key = 0
    for k in lv:
        key = key * 3 + int(k)
    return key


def behaviour_stats(pairs) -> dict:
    """Conditional entropy of action given sensory state, in bits.

    2 actions' worth of entropy would be log2(3) = 1.585, which is what a
    brain that ignores its senses produces. Anything below that means some
    sensory state has started to predict what it is about to do.
    """
    from collections import Counter, defaultdict
    import math

    groups = defaultdict(Counter)
    for s, a in pairs:
        groups[s][a] += 1
    total = max(1, len(pairs))
    h = 0.0
    for s, c in groups.items():
        n = sum(c.values())
        p = n / total
        ha = -sum((v / n) * math.log2(v / n) for v in c.values())
        h += p * ha
    acts = Counter(a for _, a in pairs)
    return {
        "cond_entropy_bits": round(h, 3),
        "max_entropy_bits": round(math.log2(3), 3),
        "n_states_seen": len(groups),
        "action_counts": [int(acts.get(i, 0)) for i in range(3)],
    }


def build_brain(neurons: int, fan_in: int, sensors: int, seed: int,
                learn: bool) -> Brain:
    b = Brain(neurons=neurons, inputs=sensors, outputs=3, fan_in=fan_in,
              sensor_span=100, seed=seed, wta_k=max(20, neurons // 400),
              target_hz=20.0)
    if learn:
        # Three-factor: a synapse tags itself when its two cells fire together,
        # and dopamine cashes the tag in when food arrives. Nothing is
        # backpropagated; the reward is one scalar that reaches every synapse.
        b.add_rule(Eligibility(decay=0.99, rate=0.004))
        b.add_rule(Dopamine())
    return b


def play(neurons: int, fan_in: int, episodes: int, seed: int, learn: bool,
         steps: int, explore: float, out_path: Optional[str],
         record: bool, warmup: int = 1500) -> dict:
    snake = Snake(size=14, view=11, seed=seed + 1)
    brain = build_brain(neurons, fan_in, snake.n_sensors, seed, learn)
    t0 = time.perf_counter()
    brain.run(warmup)
    warm = time.perf_counter() - t0
    player = SnakePlayer(brain, snake, steps_per_decision=steps,
                         explore=explore, seed=seed)

    frames = []
    scores, lengths, rates = [], [], []
    pairs = []
    t0 = time.perf_counter()
    for ep in range(episodes):
        sensor = snake.reset()
        steps_taken = 0
        while snake.alive:
            sensor, counts, action, reward, done, surprise = player.step(learn)
            steps_taken += 1
            pairs.append((state_key(sensor), int(action)))
            if record:
                frames.append({
                    "ep": ep,
                    "x": [p[0] for p in snake.body],
                    "y": [p[1] for p in snake.body],
                    "food": list(snake.food) if snake.food else None,
                    "a": int(action),
                    "r": round(float(reward), 4),
                    "e": round(float(getattr(brain, "pred_error_ema", 0.0)), 4),
                    "n": int(len(brain._prev_spikes)),
                    "ate": int(snake.ate),
                    "v": [round(float(x), 1) for x in counts],
                    "sp": [int(i) for i in brain._prev_spikes],
                })
            if done:
                break
        scores.append(float(snake.ate))
        lengths.append(steps_taken)
        rates.append(float(brain.rate_hz))
    elapsed = time.perf_counter() - t0

    result = {
        "board_size": snake.size,
        "neurons": neurons,
        "synapses": neurons * fan_in,
        "learn": bool(learn),
        "episodes": episodes,
        "score_mean": float(np.mean(scores)),
        "score_max": float(np.max(scores)),
        "score_all": scores,
        "length_mean": float(np.mean(lengths)),
        "rate_hz": float(np.mean(rates)),
        "seconds": round(elapsed, 1),
        "seconds_per_episode": round(elapsed / max(1, episodes), 2),
        "warmup_seconds": round(warm, 1),
    }
    result.update(behaviour_stats(pairs))

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        payload = {"meta": result, "frames": frames}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        result["record_path"] = out_path
        result["frames"] = len(frames)
        result["record_mb"] = round(os.path.getsize(out_path) / 1024 ** 2, 2)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neurons", type=int, default=139_000)
    ap.add_argument("--fan-in", type=int, default=360)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--explore", type=float, default=0.1)
    ap.add_argument("--no-learn", action="store_true")
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    r = play(args.neurons, args.fan_in, args.episodes, args.seed,
             not args.no_learn, args.steps, args.explore, args.out,
             args.record, args.warmup)
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
