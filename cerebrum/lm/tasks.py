"""A task the architecture is actually built for: delayed recall.

Character-level language modelling is a bad showcase for a reservoir. A
bigram table is a near-optimal estimator at small data budgets, it scales
with data, and it needs the whole corpus in memory — so a fixed random pool
plus an online linear readout is being asked to beat a specialised estimator
on that estimator's home ground.

Delayed recall is the opposite. The sequence is

    symbol, D blanks, cue, <answer>

and the only way to answer is to hold the symbol across D steps. A bigram
sees a cue and nothing else, so its best possible answer is the marginal
distribution over symbols. A reservoir that integrates has the symbol in its
state. This is the measurement that says whether the pool is doing the one
thing a liquid state machine is supposed to do.

Reported: bits and accuracy **at the answer position only**, against a bigram
fitted on the same stream. Uniform is log2(8) = 3 bits.
"""

from __future__ import annotations

import numpy as np

from .model import ReservoirLM

N_SYMBOLS = 8
BLANK = 8
CUE = 9
END = 10
N_CHARS = 11


def make_stream(trials: int, delay: int, seed: int = 0) -> tuple:
    """Build the sequence and index every answer position."""
    rng = np.random.default_rng(seed)
    seq: list = []
    answer_at: list = []
    for _ in range(trials):
        s = int(rng.integers(0, N_SYMBOLS))
        seq.append(s)
        seq.extend([BLANK] * delay)
        seq.append(CUE)
        answer_at.append(len(seq))     # the token that follows the cue
        seq.append(s)
        seq.append(END)
    return np.asarray(seq, np.int32), np.asarray(answer_at, np.int64)


def bigram_at_answers(stream: np.ndarray, answer_at: np.ndarray,
                      n_chars: int = N_CHARS) -> tuple:
    """Bigram probability assigned to the correct answer, at answer positions.

    This is the strongest counting model available — it is given the whole
    stream in advance. It cannot do better than P(symbol) because the cue
    carries no information about which symbol was shown.
    """
    counts = np.ones((n_chars, n_chars), np.float64)
    for a, b in zip(stream[:-1], stream[1:]):
        counts[a, b] += 1.0
    probs = counts / counts.sum(axis=1, keepdims=True)
    bits, correct = 0.0, 0
    for pos in answer_at:
        cue = stream[pos - 1]
        target = stream[pos]
        p = probs[cue]
        bits += -np.log2(p[target] + 1e-12)
        correct += int(np.argmax(p) == target)
    return bits / len(answer_at), correct / len(answer_at)


def run(n_neurons: int = 100_000, delay: int = 6, trials: int = 500,
        *, steps_per_token: int = 3, k_in: int = 600, n_bins: int = 256,
        k_context: int = 3, lr: float = 0.25, seed: int = 0) -> dict:
    """Train online for one pass, then measure at the answer positions."""
    stream, answer_at = make_stream(trials, delay, seed)
    model = ReservoirLM(
        n_neurons=n_neurons, n_chars=N_CHARS, k_context=k_context,
        k_in=k_in, n_bins=n_bins, steps_per_token=steps_per_token, seed=seed,
    )

    learn_until = int(len(answer_at) * 0.75)
    cut = int(answer_at[learn_until])

    for t in range(k_context, cut):
        ctx = stream[t - k_context:t]
        feats = model.run_pool(ctx, int(stream[t]))
        model.update(feats, int(stream[t + 1]), lr=lr)

    bits, correct, n = 0.0, 0, 0
    for pos in answer_at[learn_until:]:
        # The context window must match training exactly: training feeds
        # stream[t-k:t] and the current character separately, so the window
        # here has to *end before* the current character. Passing
        # stream[pos-k:pos] shifts it by one and hands the model an input it
        # has never seen — measured, that alone took accuracy to 0% and the
        # loss above the uniform baseline.
        ctx = stream[pos - 1 - k_context:pos - 1]
        p = model.predict(ctx, int(stream[pos - 1]))
        target = int(stream[pos])
        bits += -np.log2(p[target] + 1e-12)
        correct += int(np.argmax(p) == target)
        n += 1

    base_bits, base_acc = bigram_at_answers(stream, answer_at[learn_until:])
    return {
        "neurons": n_neurons,
        "delay": delay,
        "uniform_bits": float(np.log2(N_SYMBOLS)),
        "bigram_bits": float(base_bits),
        "bigram_acc": float(base_acc),
        "model_bits": bits / n,
        "model_acc": correct / n,
        "channel_bits": float(bits / n) - float(base_bits),
    }


def main() -> int:
    print("delayed recall: symbol, D blanks, cue -> which symbol?\n")
    print("neurons   delay | bigram acc | model acc | bigram bits | model bits")
    for delay in (3, 6, 10):
        r = run(n_neurons=100_000, delay=delay, trials=500, seed=0)
        print("%7d   %3d   | %9.1f%% | %8.1f%% | %10.2f | %9.2f"
              % (r["neurons"], r["delay"], r["bigram_acc"] * 100,
                 r["model_acc"] * 100, r["bigram_bits"], r["model_bits"]))
    print("\nuniform = %.2f bits; a model at chance scores 12.5%%"
          % np.log2(N_SYMBOLS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
