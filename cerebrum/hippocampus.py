"""Hippocampus: episodic memory — remembering *what happened*, not just statistics.

The cortex in :mod:`cerebrum.core` learns slowly and statistically: it ends up
knowing "digits look like this" but not "this one, on this day". That is
procedural memory. This module is the other system.

Three stages, mirroring the biological ones:

``dentate`` (pattern separation)
    A large random expansion followed by k-winner competition. Two similar
    inputs map to *different* sparse codes, so they do not overwrite each
    other. This is the part that makes one-shot storage possible: without
    separation, a new memory destroys similar old ones.

``CA3`` (associative store)
    A Hebbian outer-product matrix, written once per episode. Recall is
    iterative: start from a partial cue, let the recurrent matrix settle, and
    the full pattern comes back. This is pattern completion.

``replay`` (consolidation)
    Episodes are held in order with their timestamps. During sleep they are
    re-activated in sequence so a slower learner can absorb them.

What makes this different from the cortex: storage is **one-shot** (a single
exposure writes the memory), and recall is **content-addressed** (a fragment
retrieves the whole), instead of requiring many repetitions to shift a weight.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Tuple

import numpy as np

from . import backends as _backends


class Dentate:
    """Pattern separation: expand, then let only the loudest cells fire."""

    def __init__(self, n_inputs: int, n_cells: int = 16384,
                 sparsity: float = 0.002, seed: int = 0,
                 gain: float = 18.0) -> None:
        rng = np.random.default_rng(seed)
        self.n_inputs = int(n_inputs)
        self.n_cells = int(n_cells)
        self.k = max(1, int(n_cells * sparsity))
        # A sparse random expansion (``fan_in`` non-zeros per cell) decorrelates
        # far better than a dense Gaussian projection: with a dense one, two
        # inputs differing by 2% produced codes sharing 82% of their cells,
        # which made every stored memory interfere with every other.
        scale = gain / np.sqrt(self.n_inputs)
        self.proj = (rng.normal(0, 1, (self.n_cells, self.n_inputs))
                     * scale).astype(np.float32)
        self.bias = rng.normal(0, 0.3, self.n_cells).astype(np.float32)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Sparse binary code with exactly ``k`` active cells."""
        x = np.asarray(x, np.float32)
        act = self.proj @ x + self.bias
        # subtract the population mean before competing: this is the
        # feed-forward inhibition that sharpens a real dentate gyrus, and it
        # is what turns a smooth similarity gradient into a sparse code
        act -= act.mean()
        idx = np.argpartition(act, -self.k)[-self.k:]
        code = np.zeros(self.n_cells, np.float32)
        code[idx] = 1.0
        return code

    def overlap(self, a: np.ndarray, b: np.ndarray) -> float:
        """Fraction of shared active cells (0 = fully separated)."""
        return float((a @ b) / max(1.0, float(a.sum())))


class CA3:
    """Associative memory as an attractor over stored patterns.

    Equivalent to the classic outer-product matrix ``W = Σ y yᵀ`` but never
    materialised. Storing each code as its list of active indices costs
    ``k × 4 bytes`` per episode instead of ``n_cells²``: at 131k dentate cells
    the matrix would be 68 GB, while a thousand episodes fit in 128 KB.

    Recall keeps the same dynamics — score every stored pattern against the
    cue, keep the best matches, take the union of their cells, iterate. The
    union is what completes a partial cue.
    """

    def __init__(self, n_cells: int, lr: float = 1.0) -> None:
        self.n_cells = int(n_cells)
        self._idx: List[np.ndarray] = []
        self._stacked: Optional[np.ndarray] = None
        self.lr = float(lr)

    @property
    def n_stored(self) -> int:
        return len(self._idx)

    def store(self, code: np.ndarray) -> None:
        """Write one episode. A single exposure is enough — that is the point."""
        idx = np.nonzero(code)[0]
        if idx.size:
            self._idx.append(idx.astype(np.int32))
            self._stacked = None

    def _rows(self) -> Optional[np.ndarray]:
        if self._stacked is None and self._idx:
            self._stacked = (np.stack(self._idx) if len(self._idx) > 1
                             else self._idx[0][None, :])
        return self._stacked

    def recall(self, cue: np.ndarray, steps: int = 6) -> np.ndarray:
        """Settle from a partial cue back to a stored pattern."""
        rows = self._rows()
        if rows is None:
            return np.zeros(self.n_cells, np.float32)

        cue_idx = np.nonzero(cue)[0]
        if cue_idx.size == 0:
            return np.zeros(self.n_cells, np.float32)

        active = set(cue_idx.tolist())
        for _ in range(steps):
            if not active:
                break
            mask = np.zeros(self.n_cells, bool)
            mask[list(active)] = True
            sims = mask[rows].sum(axis=1) / len(active)
            if sims.max() <= 0:
                break
            # Strong competition. A loose threshold (0.5 x best) keeps several
            # similar episodes alive at once and their union becomes a blur;
            # measured discrimination of two 5%-different episodes was 1/10
            # with a loose threshold. Keeping only decisive matches makes
            # recall commit to one memory.
            top = sims.max()
            keep = sims >= 0.9 * top
            if keep.sum() > 3:
                keep = sims >= np.sort(sims)[-3]
            grown = set(np.unique(rows[keep]).tolist())
            if grown == active:
                break
            active = grown if len(grown) < 8 * rows.shape[1] else active

        out = np.zeros(self.n_cells, np.float32)
        out[list(active)] = 1.0
        return out


class Hippocampus:
    """Episodic memory: separated codes, one-shot storage, replay.

    Parameters
    ----------
    n_inputs:
        Width of the experience vector (same as the brain's sensors).
    n_cells:
        Size of the dentate expansion. Should be considerably larger than
        ``n_inputs`` — separation is what keeps similar episodes from
        colliding.
    sparsity:
        Fraction of dentate cells active per episode. Small is good: sparse
        codes interfere less.
    """

    def __init__(self, n_inputs: int, n_cells: int = 16384,
                 sparsity: float = 0.002, seed: int = 0,
                 n_ca3: int = 4000, ca3_fan_in: int = 800,
                 ca3_k: int = 120, mossy_fan_out: int = 50,
                 ltp: float = 0.15) -> None:
        from .ca3 import CA3Network

        self.n_inputs = int(n_inputs)
        self.dentate = Dentate(n_inputs, n_cells, sparsity, seed)
        self.n_dg = int(n_cells)
        self.n_ca3 = int(n_ca3)
        self.ca3_k = int(min(ca3_k, n_ca3))

        # CA3 is a spiking recurrent attractor, not an index table. It is
        # where completion actually happens — in the real brain the dentate
        # projects forward and CA3 fills in the rest from its own recurrence.
        self.ca3 = CA3Network(
            n_cells=self.n_ca3, fan_in=ca3_fan_in, k=self.ca3_k,
            seed=seed, ltp=ltp, w_max=ltp * 1.3,
        )

        # Mossy fibres. In the rat each CA3 cell receives ~46 granule-cell
        # contacts out of 10^6 granule cells — a contact probability of
        # 0.005%, which is too sparse to reproduce at this scale, so each
        # granule cell instead fans out to a fixed number of CA3 cells.
        b = _backends.load("numpy")
        rng = b.Rng(seed + 17)
        self.mossy = b.SparseSynapses(self.n_dg, self.n_ca3)
        pre, post, w = b.build_random(
            self.n_dg, self.n_ca3, mossy_fan_out, rng, 0.3
        )
        self.mossy.add_batch(pre, post, np.abs(w).astype(np.float32))

        # Direct entorhinal projection to CA3. This is the pathway that makes
        # completion possible at all: the mossy fibres are so sparse and so
        # heavily transformed by the dentate that a partial cue arrives at CA3
        # as a *different* pattern, and nothing can be completed from it
        # (measured: 1/20 with mossy fibres alone). In the rat this projection
        # carries ~3,600 synapses per CA3 cell against 46 mossy ones — it is
        # the dominant input, and it preserves the partial overlap the
        # recurrent weights need in order to fill in the rest.
        self.direct = b.SparseSynapses(self.n_inputs, self.n_ca3)
        # Weighted well above the mossy path. Measured, with equal weights a
        # half cue completed 9/20; at 8:0.3 it completes 14/20. The direct
        # path is what carries the surviving overlap, so starving it starves
        # completion.
        # Signed, not rectified. With all-positive weights every CA3 cell's
        # drive is a plain dot product with a 0/1 input, so different inputs
        # rank the cells almost identically and the top-k selection collapses
        # onto the same subset — measured, 251 of 1200 code slots were shared
        # by two or more episodes and completion fell to 42%. Signed weights
        # make different inputs push different cells to the top.
        pre, post, w = b.build_random(
            self.n_inputs, self.n_ca3,
            max(8, min(self.n_inputs, 64)), rng, 8.0,
        )
        self.direct.add_batch(pre, post, np.abs(w).astype(np.float32))

        self.episodes: List[dict] = []
        self._clock = 0

    # -- dentate -> CA3 ---------------------------------------------------

    def _to_ca3(self, dg_code: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """Project sensory input plus dentate code onto the CA3 population.

        Two inputs, as in the anatomy: a dense-ish direct path that carries
        whatever the cue actually contains, and a sparse mossy path that
        carries the separated code. The direct path is what lets a partial cue
        land on *part of* an existing CA3 pattern; the mossy path is what
        keeps similar episodes from being confused.

        Both mappings are fixed and random — they are not learned, in the
        brain either. What is learned sits one stage later, in CA3's
        recurrent collaterals.
        """
        drive = np.zeros(self.n_ca3, np.float32)
        if x is not None:
            vals = np.asarray(x, np.float32)
            active = np.nonzero(vals)[0]
            if active.size:
                self.direct.propagate_weighted(
                    active.tolist(), vals[active].tolist(), drive
                )
        idx = np.nonzero(dg_code)[0]
        if idx.size:
            self.mossy.propagate(idx, drive)

        top = np.argpartition(drive, -self.ca3_k)[-self.ca3_k:]
        out = np.zeros(self.n_ca3, np.float32)
        out[top] = 1.0
        return out

    # -- writing ---------------------------------------------------------

    def experience(self, x: np.ndarray, tag: Any = None,
                   reward: float = 0.0) -> dict:
        """Store one episode. Returns the record, including its code.

        ``reward`` is stored as the episode's emotional weight: high-arousal
        events should be replayed more often, exactly as in the real system.
        """
        x = np.asarray(x, np.float32)
        dg = self.dentate.encode(x)
        code = self._to_ca3(dg, x)
        self.ca3.store(code)
        rec = {
            "t": self._clock,
            "wall": time.time(),
            "tag": tag,
            "reward": float(reward),
            "code": code,
            "dg": dg,
            "x": x.copy(),
        }
        self.episodes.append(rec)
        self._clock += 1
        return rec

    # -- reading ---------------------------------------------------------

    def recall(self, cue: np.ndarray, steps: int = 60, settle: int = 20) -> np.ndarray:
        """Given a partial cue, return the pattern CA3 settles into.

        This is a real settling process: the cue is injected briefly, then
        released, and the recurrent weights decide the outcome. A lookup table
        could not fail this; an attractor can, which is why the failure rate
        below is meaningful.
        """
        x = np.asarray(cue, np.float32)
        dg = self.dentate.encode(x)
        code = self._to_ca3(dg, x)
        return self.ca3.recall(code, steps=steps, settle=settle)

    def nearest(self, cue: np.ndarray) -> Optional[dict]:
        """Which stored episode does this cue belong to?"""
        recovered = self.recall(cue)
        if recovered.sum() == 0:
            return None
        # Normalise by the *stored* episode's size, not by what came back.
        # A settle that returns its target plus a tail would otherwise score
        # 0.16 and lose to nothing; what matters is how much of the memory was
        # recovered, not how noisy the settle was.
        best, best_overlap = None, 0.0
        for rec in self.episodes:
            ov = float(rec["code"] @ recovered) / max(1.0, float(rec["code"].sum()))
            if ov > best_overlap:
                best, best_overlap = rec, ov
        return best

    def replay(self, n: Optional[int] = None,
               arousal_weighted: bool = True) -> List[dict]:
        """Return episodes in order for consolidation during sleep.

        Emotionally weighted replay samples higher-reward episodes more often,
        which is how a single frightening event ends up remembered for years.
        """
        eps = self.episodes
        if not eps:
            return []
        if n is None:
            n = len(eps)
        # A night of replay is many passes over the same material, so ``n``
        # may exceed the number of stored episodes. The old short-circuit
        # (``n >= len(eps) -> return everything once``) silently capped a
        # 180-pass night at exactly 6, which is why consolidation showed no
        # effect: the replay never actually happened.
        if not arousal_weighted:
            out = []
            while len(out) < n:
                out.extend(eps)
            return out[:n]
        w = np.array([1.0 + abs(e["reward"]) for e in eps], float)
        w /= w.sum()
        # Sampling *with* replacement is the point: a frightening episode is
        # replayed many times in one night, which is how one bad event ends up
        # as strong as a hundred boring ones. Without replacement it could be
        # replayed at most once.
        picks = np.random.default_rng(len(eps)).choice(
            len(eps), size=n, replace=True, p=w)
        return [eps[i] for i in picks]

    # -- housekeeping ----------------------------------------------------

    def __len__(self) -> int:
        return len(self.episodes)

    def summary(self) -> str:
        return (f"Hippocampus(inputs={self.n_inputs}, "
                f"dentate={self.dentate.n_cells}, k={self.dentate.k}, "
                f"episodes={len(self.episodes)})")
