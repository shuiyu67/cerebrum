"""Character-level corpus loading.

No download: the machine has no route to GitHub and 100 MB of disk. The
corpus is this repository's own source — 370k characters of Python and
Markdown. A character LM on code is a well-defined task with real structure
(indentation, keywords, punctuation), and unlike a toy alphabet it cannot be
solved by memorising 26 transition probabilities.
"""

from __future__ import annotations

import io
import os
from typing import Optional

import numpy as np


CORPUS_PATH = "data/corpus.txt"


def build_corpus_from_sources(root: str = ".", out: str = CORPUS_PATH) -> str:
    """Assemble the corpus from this repository's own source files.

    The corpus is deliberately not committed: it is derived from the tree, so
    it would double the repository and go stale the moment a file changed.
    Building it on demand is what makes a fresh clone runnable — without
    this, `python -m cerebrum.lm.train` and the LM tests both fail on a
    machine that has never run them.
    """
    import glob
    import os

    files = sorted(
        glob.glob(os.path.join(root, "cerebrum", "**", "*.py"), recursive=True)
        + glob.glob(os.path.join(root, "tests", "*.py"))
        + glob.glob(os.path.join(root, "*.md"))
    )
    text = "\n".join(io.open(f, encoding="utf-8", errors="ignore").read()
                      for f in files)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def load_text(path: str) -> str:
    if not os.path.exists(path):
        return build_corpus_from_sources(out=path)
    with open(path, encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def build_vocab(text: str, min_count: int = 120) -> tuple:
    """Character vocabulary, with everything rare folded into one symbol.

    Keeping all 311 distinct characters would put 8.3 bits of uniform
    entropy on the task and spend most of it on characters that appear twice.
    Characters below ``min_count`` become a single placeholder.
    """
    counts: dict = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    keep = sorted([c for c, k in counts.items() if k >= min_count])
    chars = keep + ["\x00"]              # \x00 is the rare-character bucket
    stoi = {c: i for i, c in enumerate(chars)}
    unknown = len(chars) - 1
    return chars, stoi, unknown


def encode(text: str, stoi: dict, unknown: int) -> np.ndarray:
    return np.fromiter((stoi.get(ch, unknown) for ch in text),
                       dtype=np.int32, count=len(text))


class Corpus:
    """Encoded stream plus contiguous train / validation splits."""

    def __init__(self, path: str = CORPUS_PATH, *, min_count: int = 120,
                 val_frac: float = 0.1) -> None:
        text = load_text(path)
        self.chars, self.stoi, self.unknown = build_vocab(text, min_count)
        self.itos = {i: c for c, i in self.stoi.items()}
        data = encode(text, self.stoi, self.unknown)
        # Blocked interleaved split, not a tail split. The corpus is a
        # concatenation of files in alphabetical order, so "the last 10%" is
        # the README — Markdown prose against Python source. A tail split
        # therefore measures a distribution shift, not generalisation:
        # measured, held-out loss oscillated between 4.3 and 5.4 bits while
        # training loss fell smoothly.
        block = 200
        blocks = [data[i:i + block] for i in range(0, len(data), block)]
        every = max(2, int(round(1.0 / max(val_frac, 1e-3))))
        val_blocks = blocks[every - 1::every]
        train_blocks = [b for i, b in enumerate(blocks) if i % every != every - 1]
        self.train = np.concatenate(train_blocks) if train_blocks else data
        self.val = np.concatenate(val_blocks) if val_blocks else data[-block:]
        self.n_chars = len(self.chars)
        self.text_len = len(data)

    @property
    def uniform_bits(self) -> float:
        return float(np.log2(self.n_chars))

    def bigram_bits(self, train: Optional[np.ndarray] = None,
                    eval_on: Optional[np.ndarray] = None) -> float:
        """Bigram baseline, by default fitted on the *whole* training split.
        Callers that train on a slice should pass that slice, or the
        comparison gives the baseline 40x more data than the model."""
        """Bigram baseline with add-one smoothing.

        This is the number the reservoir has to beat. A recurrent pool that
        cannot beat "predict the next character from the current one" is not
        doing anything the context is not already doing.
        """
        train = self.train if train is None else train
        eval_on = self.val if eval_on is None else eval_on
        v = self.n_chars
        counts = np.ones((v, v), np.float64)
        for a, b in zip(train[:-1], train[1:]):
            counts[a, b] += 1.0
        probs = counts / counts.sum(axis=1, keepdims=True)
        bits = -np.log2(probs[eval_on[:-1], eval_on[1:]] + 1e-12)
        return float(bits.mean())

    def summary(self) -> str:
        return (f"Corpus({self.text_len:,} chars, vocab {self.n_chars}, "
                f"train {len(self.train):,} / val {len(self.val):,}, "
                f"uniform {self.uniform_bits:.2f} bits)")
