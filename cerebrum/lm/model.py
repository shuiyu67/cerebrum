"""The language model: a fixed million-neuron pool plus a trained readout.

Architecture
------------
``context`` (the last K characters) -> sparse input neurons in the pool ->
K milliseconds of recurrent spiking -> a binned state vector -> a linear
readout -> a distribution over the next character.

Only the readout learns. The pool's synapses are random and frozen, which is
the whole point: the claim under test is that a million spiking neurons with
no training at all still provide a representation rich enough for a linear
layer to predict the next character better than the character itself does.

The readout also gets the current character as a direct input. That is a skip
connection, and it is deliberate: it makes the bigram statistics trivially
available, so whatever the model gains beyond the bigram baseline has to come
from the recurrent pool. Without it the comparison would confound two
effects.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .reservoir import SparseReservoir


class ReservoirLM:
    """Reservoir language model with a softmax readout.

    Parameters
    ----------
    n_neurons:
        Pool size.
    n_chars:
        Vocabulary size.
    k_context:
        Characters of context, one input group per position.
    k_in:
        Pool neurons activated per (position, character) pair.
    n_bins:
        Width of the state vector the readout sees.
    steps_per_token:
        Milliseconds of recurrent dynamics per character.
    """

    def __init__(
        self,
        n_neurons: int = 1_000_000,
        n_chars: int = 96,
        *,
        k_context: int = 6,
        k_in: int = 1200,
        n_bins: int = 4096,
        whiten: bool = False,
        steps_per_token: int = 4,
        drive: float = 0.5,
        seed: int = 0,
        **reservoir_kwargs,
    ) -> None:
        self.n_chars = int(n_chars)
        self.k_context = int(k_context)
        self.k_in = int(k_in)
        self.n_bins = int(n_bins)
        self.whiten = bool(whiten)
        self._f_mean = np.zeros(self.n_bins, np.float32)
        self._f_var = np.ones(self.n_bins, np.float32)
        self.steps_per_token = int(steps_per_token)
        self.drive = float(drive)
        self.n_neurons = int(n_neurons)

        self.pool = SparseReservoir(n_neurons, seed=seed, **reservoir_kwargs)

        rng = np.random.default_rng(seed + 1)
        # One random group of pool neurons per (position, character). Fixed
        # for the life of the model: this is the input wiring, not a weight
        # that learning touches.
        self.embed = np.empty((self.k_context, self.n_chars, self.k_in),
                              np.int32)
        for pos in range(self.k_context):
            for ch in range(self.n_chars):
                self.embed[pos, ch] = rng.choice(
                    n_neurons, self.k_in, replace=False)

        # Readout: state + skip connection -> next-character scores, started
        # at zero. Random initial weights would give a confident wrong prior
        # and the first few hundred updates would be spent undoing it.
        self.n_direct = self.n_chars * (self.k_context + 1)
        d_in = self.n_bins + self.n_direct
        self.W = np.zeros((self.n_chars, d_in), np.float32)
        self.d_in = d_in
        self.loss_ema = float(np.log(n_chars))
        self.updates = 0

    # -- features --------------------------------------------------------

    def encode(self, context: np.ndarray) -> np.ndarray:
        """Context characters -> the pool neurons they drive."""
        parts = [self.embed[i, int(ch)]
                 for i, ch in enumerate(context[-self.k_context:])]
        return np.concatenate(parts)

    def run_pool(self, context: np.ndarray, current: int) -> np.ndarray:
        """Drive the pool for ``steps_per_token`` and return its state."""
        ext = self.encode(context)
        # The pool is not reset between characters. Its state *is* the
        # memory of the preceding context; clearing it every token would
        # leave each prediction depending on the current character alone,
        # which is exactly the bigram model the reservoir is supposed to
        # beat.
        acc = np.zeros(self.n_bins, np.float32)
        for _ in range(self.steps_per_token):
            idx = self.pool.step(ext, ext_val=self.drive)
            if idx.size:
                acc += self.pool.binned_state(idx, self.n_bins)
        acc /= float(self.steps_per_token)
        # Unit-norm state. Without this the readout diverges on the first few
        # updates: the state has ~500 active bins at magnitude 1-4, a norm
        # near 90, so a learning rate of 0.08 moves a weight by 7 and the
        # loss goes from 4.3 (uniform) to 24 within a thousand characters.
        norm = float(np.linalg.norm(acc)) + 1e-8
        state = (acc / norm).astype(np.float32)
        # Running whitening, off by default. It was tried because the 1024
        # projection dimensions have very different variances, and it made
        # things worse in every configuration tested (4.59 -> 5.28 bits at a
        # 20k pool): the variance estimate lags the features it is meant to
        # normalise, and clipping the standardised value at +-4 throws away
        # exactly the rare large responses that carry context.
        if self.whiten:
            m = 0.001
            self._f_mean += (state - self._f_mean) * m
            self._f_var += ((state - self._f_mean) ** 2 - self._f_var) * m
            z = (state - self._f_mean) / np.sqrt(self._f_var + 0.05)
            state = np.clip(z, -4.0, 4.0).astype(np.float32)
        # Direct features: the current character and the context, one-hot.
        # Without these the readout has to recover "the last character" from
        # a binned spike count, which it cannot do exactly, and the model
        # ends up worse than a bigram table it should trivially contain.
        # Keeping them makes the comparison meaningful: whatever the model
        # gains over an n-gram has to come from the pool.
        direct = np.zeros(self.n_chars * (self.k_context + 1), np.float32)
        for i, ch in enumerate(context[-self.k_context:]):
            direct[i * self.n_chars + int(ch)] = 1.0
        direct[self.k_context * self.n_chars + int(current)] = 1.0
        direct /= float(np.sqrt(self.k_context + 1))
        return np.concatenate([state, direct])

    # -- inference -------------------------------------------------------

    def logits(self, features: np.ndarray) -> np.ndarray:
        return self.W @ features

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def predict(self, context: np.ndarray, current: int) -> np.ndarray:
        return self.softmax(self.logits(self.run_pool(context, current)))

    # -- learning --------------------------------------------------------

    def update(self, features: np.ndarray, target: int,
               lr: float = 0.05, weight_decay: float = 1e-5) -> float:
        """One softmax cross-entropy step on the readout.

        This is the only place in the whole engine where a gradient exists,
        and it is one linear layer — the same amount of gradient a logistic
        regression uses. The million neurons have none.
        """
        z = self.logits(features)
        p = self.softmax(z)
        loss = float(-np.log(p[int(target)] + 1e-12))
        grad = p
        grad[int(target)] -= 1.0
        self.W -= (lr * np.outer(grad, features)).astype(np.float32)
        if weight_decay:
            self.W *= np.float32(1.0 - lr * weight_decay)
        self.updates += 1
        self.loss_ema = 0.99 * self.loss_ema + 0.01 * loss
        return loss

    # -- evaluation ------------------------------------------------------

    def evaluate(self, stream: np.ndarray, limit: int = 2000) -> dict:
        """Bits per character on a stream, and top-1 accuracy."""
        bits, correct, n = 0.0, 0, 0
        start = self.k_context
        stop = min(len(stream) - 1, start + limit)
        for t in range(start, stop):
            ctx = stream[t - self.k_context:t]
            feats = self.run_pool(ctx, int(stream[t]))
            p = self.softmax(self.logits(feats))
            tgt = int(stream[t + 1])
            bits += -np.log2(p[tgt] + 1e-12)
            correct += int(np.argmax(p) == tgt)
            n += 1
        return {"bits_per_char": bits / max(1, n),
                "accuracy": correct / max(1, n),
                "tokens": n}

    def summary(self) -> str:
        return (f"ReservoirLM({self.pool.summary()}, context {self.k_context}, "
                f"state {self.n_bins}+{self.n_direct}, "
                f"readout {self.W.size:,} weights)")
