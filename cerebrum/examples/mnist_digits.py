"""MNIST: can the self-organising brain learn to classify images?

Division of labour, following the "teacher and doctor" rule:

* **teacher** — shows an image, then says whether the answer was right;
* **doctor** — competition (k-WTA) and homeostasis keep the population in a
  healthy firing band;
* **the brain** — nothing about its internal layout is specified. Which neurons
  respond to which digit is decided by a local Hebbian rule on its input
  synapses. Nobody assigns neurons to digits.

The readout is a small set of class neurons whose connections are adjusted only
by the teacher's verdict — a three-factor update: presynaptic activity ×
(correct minus spoken) × reward. That is the grading, not the learning of
features.

Run::

    python -m cerebrum.examples.mnist_digits --neurons 2000 --per-class 2000
"""

from __future__ import annotations

import argparse
import gzip
import os
import struct
import time

import numpy as np

from ..core import Brain


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def _read_idx(path: str, magic_expected: int) -> np.ndarray:
    # IDX header: a 4-byte magic whose low byte *is* the dimension count
    # (2051 = 0x0803 -> uint8, 3 dimensions). There is no separate ndim field;
    # reading one shifts everything and the array shape becomes garbage.
    with gzip.open(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        if magic != magic_expected:
            raise ValueError(f"{path}: unexpected magic {magic}")
        ndim = magic & 0xFF
        dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        data = np.frombuffer(f.read(), np.uint8)
    return data.reshape(tuple(int(d) for d in dims))


MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"

FILES = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")


def ensure_data() -> None:
    """Fetch MNIST into ``data/`` if it is not there yet.

    Without this a fresh clone fails with a bare FileNotFoundError, which is a
    bad first five seconds for anyone who just wanted to see it run.
    """
    import urllib.request

    os.makedirs(DATA_DIR, exist_ok=True)
    for name in FILES:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(MNIST_BASE + name, path)


def load_mnist(downsample: int = 2):
    ensure_data()
    """Return (train_x, train_y, test_x, test_y) with inputs in [0, 1]."""
    tr_x = _read_idx(os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz"), 2051)
    tr_y = _read_idx(os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz"), 2049)
    te_x = _read_idx(os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz"), 2051)
    te_y = _read_idx(os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz"), 2049)

    def prep(x):
        if downsample > 1:
            n, h, w = x.shape
            x = x.reshape(n, h // downsample, downsample,
                          w // downsample, downsample).mean(axis=(2, 4))
        return (x.reshape(len(x), -1).astype(np.float32) / 255.0)

    return prep(tr_x), tr_y.astype(np.int64), prep(te_x), te_y.astype(np.int64)


# ---------------------------------------------------------------------------
# readout: the teacher's grading
# ---------------------------------------------------------------------------

class Readout:
    """Class neurons driven by population activity; adjusted by the verdict.

    ``score[c] = W[c] · rate``. On a mistake the true class is pushed toward
    the current pattern and the wrong one pushed away. This is a perceptron
    with a spiking feature vector, and it exists to grade the brain's own
    features — it does not create them.
    """

    def __init__(self, n_classes: int, n_features: int, lr: float = 0.05,
                 seed: int | None = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, (n_classes, n_features)).astype(np.float32)
        self.lr = float(lr)
        self.n_classes = n_classes

    def predict(self, rate: np.ndarray) -> int:
        return int(np.argmax(self.W @ rate))

    def learn(self, rate: np.ndarray, label: int, correct: bool,
              reward: float = 1.0) -> None:
        """Reinforce the right class, penalise the wrong one."""
        norm = float(np.linalg.norm(rate)) + 1e-9
        step = self.lr * reward / norm
        pred = self.predict(rate)
        self.W[label] += step * rate
        if not correct and pred != label:
            self.W[pred] -= step * rate
        np.clip(self.W, -6.0, 6.0, out=self.W)


class RidgeReadout:
    """Least-squares grading, accumulated without storing the feature matrix.

    A perceptron leaves accuracy on the table because it only ever nudges the
    weight vector by the current sample. The same spiking features score
    83.1% under a perceptron and 93.1% under ridge regression — the readout,
    not the brain, was the bottleneck.

    ``G = Σ a aᵀ`` and ``B = Σ a yᵀ`` are accumulated as images arrive
    (``a`` is the rate vector plus a bias term), so the cost is one 2000×2000
    matrix rather than a 20000×2000 feature matrix.
    """

    def __init__(self, n_classes: int, n_features: int, lam: float = 1.0):
        d = n_features + 1
        self.G = np.zeros((d, d), np.float64)
        self.B = np.zeros((d, n_classes), np.float64)
        self.lam = float(lam)
        self.W = None
        self.n_classes = n_classes

    def observe(self, rate: np.ndarray, label: int) -> None:
        a = np.empty(len(rate) + 1, np.float64)
        a[:-1] = rate
        a[-1] = 1.0
        self.G += np.outer(a, a)
        self.B[:, label] += a

    def solve(self) -> None:
        self.W = np.linalg.solve(self.G + self.lam * np.eye(len(self.G)), self.B)

    def predict(self, rate: np.ndarray) -> int:
        if self.W is None:
            self.solve()
        a = np.empty(len(rate) + 1, np.float64)
        a[:-1] = rate
        a[-1] = 1.0
        return int(np.argmax(a @ self.W))


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------

def evaluate(brain, readout, x, y, steps, limit=None):
    ok = 0
    total = 0
    for i in range(len(x) if limit is None else min(limit, len(x))):
        rate = present(brain, x[i], steps, learn=False)
        ok += readout.predict(rate) == int(y[i])
        total += 1
    return ok / max(1, total)


def present(brain, image, steps, learn=True):
    """Show one image for ``steps`` ms and return mean firing rates."""
    acc = np.zeros(brain.n_neurons, np.float32)
    for _ in range(steps):
        brain.step(image.tolist())
        if len(brain._prev_spikes):
            acc[np.asarray(brain._prev_spikes, np.int64)] += 1.0
    brain.reset_state()
    return acc / float(steps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neurons", type=int, default=2000)
    ap.add_argument("--per-class", type=int, default=2000,
                    help="training images per class")
    ap.add_argument("--steps", type=int, default=20, help="ms per image")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--input-lr", type=float, default=0.02)
    ap.add_argument("--wta", type=int, default=40)
    ap.add_argument("--no-input-plasticity", action="store_true")
    ap.add_argument("--test-limit", type=int, default=1000)
    ap.add_argument("--readout", choices=("ridge", "perceptron"),
                    default="ridge",
                    help="how the teacher grades (ridge is much stronger)")
    args = ap.parse_args()

    tr_x, tr_y, te_x, te_y = load_mnist(args.downsample)
    n_in = tr_x.shape[1]
    print(f"MNIST: {len(tr_x)} train / {len(te_x)} test, inputs {n_in} "
          f"({int(n_in ** 0.5)}x{int(n_in ** 0.5)})")

    brain = Brain(
        neurons=args.neurons, inputs=n_in, outputs=10, fan_in=20,
        sensor_span=max(16, args.neurons * 16 // n_in),
        seed=1, wta_k=args.wta, target_hz=20.0,
        input_plasticity=not args.no_input_plasticity,
        input_lr=args.input_lr,
    )
    print(brain.summary())

    # warm the population up before any stimulus
    brain.run(1500)

    readout = (RidgeReadout(10, args.neurons)
               if args.readout == "ridge" else Readout(10, args.neurons,
                                                       lr=args.lr, seed=0))

    idx = []
    for c in range(10):
        idx.extend(np.nonzero(tr_y == c)[0][: args.per_class])
    idx = np.array(idx)
    print(f"training on {len(idx)} images ({args.per_class}/class), "
          f"{args.epochs} epoch(s)")

    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        order = np.random.default_rng(epoch).permutation(len(idx))
        for n, k in enumerate(order):
            i = idx[k]
            rate = present(brain, tr_x[i], args.steps)
            if isinstance(readout, RidgeReadout):
                readout.observe(rate, int(tr_y[i]))
            else:
                pred = readout.predict(rate)
                readout.learn(rate, int(tr_y[i]), pred == int(tr_y[i]))
            if (n + 1) % 2000 == 0:
                print(f"  epoch {epoch + 1}  {n + 1:6d} images  "
                      f"{time.perf_counter() - t0:6.0f}s  "
                      f"rate {brain.rate_hz:5.1f} Hz", flush=True)
        if isinstance(readout, RidgeReadout):
            readout.solve()
        acc = evaluate(brain, readout, te_x, te_y, args.steps,
                       limit=args.test_limit)
        print(f"after epoch {epoch + 1}: test accuracy {acc * 100:.1f}%")

    acc = evaluate(brain, readout, te_x, te_y, args.steps, limit=args.test_limit)
    print(f"\nfinal test accuracy: {acc * 100:.1f}%  "
          f"({args.test_limit} images)")
    print(f"chance: 10.0%   total wall time {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
