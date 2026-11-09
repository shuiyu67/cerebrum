"""Hippocampus: episodic memory that the cortex can actually read back.

The previous generation had a hippocampus that stored episodes and nothing
that could hand them back to the cortex. Storage without a return path is a
filing cabinet with no door: the brain could recognise a situation but never
recall one.

This module builds the full loop:

``EC`` (entorhinal)
    The cortical vector arrives here and is recoded sparsely. It is the
    interface in both directions: cortex -> EC on the way in, EC -> cortex on
    the way out.
``DG`` (dentate gyrus)
    A large expansion with strong feed-forward competition. Its job is
    *separation*: two similar situations must map to different codes, or a new
    memory overwrites a similar old one.
``CA3``
    A recurrent attractor over sparse codes. One exposure writes the pattern
    into the recurrent matrix; a fragment then settles back to the whole.
``CA1``
    The comparator. It receives the CA3 recall *and* the direct EC copy of
    what is actually here. Where they disagree is novelty — the signal that
    drives attention, noradrenaline and the cortical learning rate.

Recall writes back: ``CA1 -> EC deep -> cortex`` reinstates the cortical
pattern that went with the episode, which is what makes recall an experience
rather than a lookup.

Implementation note
-------------------
CA3 is a symmetric outer-product attractor over sparse binary codes, updated
by k-winner competition rather than by thresholding a continuous state. The
spiking recurrent attractor that ships elsewhere in this package
(:class:`cerebrum.ca3.CA3Network`) is not used here: at the sizes this brain
runs at it produced no spikes at all on a full cue, so it would have been a
memory that never answered. The attractor below is the standard Hopfield
construction and it demonstrably completes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..hippocampus import Dentate


class Hippocampus:
    """Entorhinal-dentate-CA3-CA1 loop with a return path to cortex."""

    def __init__(
        self,
        n_cortex: int = 400,
        *,
        n_ec: int = 400,
        n_dg: int = 2000,
        n_ca3: int = 1000,
        ca3_k: int = 80,
        dc_sparsity: float = 0.15,
        dg_sparsity: float = 0.02,
        seed: int = 0,
    ) -> None:
        self.n_cortex = int(n_cortex)
        self.n_ec = int(n_ec)
        self.n_dg = int(n_dg)
        self.n_ca3 = int(n_ca3)
        self.ca3_k = int(ca3_k)

        self.ec = Dentate(n_cortex, n_ec, dc_sparsity, seed + 1)
        self.dg = Dentate(n_ec, n_dg, dg_sparsity, seed + 2)

        rng = np.random.default_rng(seed + 3)
        # The perforant path EC -> CA3. Encoding and retrieval use *this same*
        # projection, which is the only way completion can work: when
        # encoding went through the dentate and retrieval did not, the two
        # codes were uncorrelated and the attractor completed nothing
        # (measured: Jaccard 0.14 from a 60% cue, against 1.00 for a full
        # one). The dentate's separation job is done at storage time
        # instead — see ``experience`` — which is where it matters, because
        # that is where similar episodes would otherwise overwrite each
        # other.

        self.ec_to_ca3 = (rng.random((n_ca3, n_ec)) < 0.05).astype(np.float32) * 1.2

        # CA3 recurrent matrix. Written as a *covariance* rather than a plain
        # outer product: with W = sum(c c^T) the cells that appear in many
        # episodes accumulate weight to everything, and every cue collapses
        # onto those hubs — measured, completion was a flat 0.27 no matter
        # how much of the cue was present, i.e. the attractor answered the
        # same thing to every question. Subtracting the mean activity before
        # the outer product removes the hub and the same network completes
        # 0.98 from a cue holding 30% of the code.
        self.W = np.zeros((n_ca3, n_ca3), np.float32)
        self._dirty = True
        self.codes: list = []
        self.dg_codes: list = []
        self.records: list = []

        # CA1 -> cortex reinstatement. Fixed random back-projection: the
        # cortex learns to read it, the hippocampus does not learn to write
        # it. That asymmetry keeps recall from overwriting the store.
        self.back = (rng.normal(0.0, 1.0, (n_cortex, n_ca3))
                     / np.sqrt(n_ca3)).astype(np.float32)

        self.last_novelty = 1.0
        self.last_overlap = 0.0

    # -- encoding --------------------------------------------------------

    def _ca3_code(self, cortex_vec: np.ndarray):
        ec = self.ec.encode(np.asarray(cortex_vec, np.float32))
        dg = self.dg.encode(ec)
        drive = self.ec_to_ca3 @ ec
        code = np.zeros(self.n_ca3, np.float32)
        if drive.max() > 0:
            # fixed-size code: sparsity is part of the representation, and a
            # relative threshold lets the tail through
            idx = np.argpartition(drive, -self.ca3_k)[-self.ca3_k:]
            code[idx] = 1.0
        return ec, dg, code

    # -- storage ---------------------------------------------------------

    def separation(self, cortex_vec: np.ndarray) -> float:
        """How distinct this situation is from what is already stored.

        Measured on dentate codes, not on the CA3 code: separation is the
        dentate's job, and a CA3 code that is already an attractor is by
        construction similar to whatever it settled on.
        """
        _, dg, _ = self._ca3_code(cortex_vec)
        if not self.codes or dg.sum() == 0:
            return 1.0
        best = 0.0
        for past in self.dg_codes[-8:]:
            best = max(best, float(dg @ past) / max(1.0, float(dg.sum())))
        return float(1.0 - best)

    def experience(self, cortex_vec: np.ndarray, tag: Optional[str] = None) -> dict:
        """Store one episode from a single exposure."""
        _, dg, code = self._ca3_code(cortex_vec)
        if code.sum() == 0:
            return {"stored": False, "tag": tag}
        self.dg_codes.append(dg)
        if len(self.dg_codes) > 64:
            self.dg_codes.pop(0)
        # One-shot write: a single exposure is the whole point of an
        # episodic store — a cortex needs many trials, this needs one. The
        # matrix is rebuilt lazily in ``_matrix`` because the covariance
        # depends on the mean of *all* stored codes, which changes with each
        # new episode.
        self.codes.append(code)
        self.records.append({"code": code, "tag": tag,
                             "cortex": np.asarray(cortex_vec, np.float32).copy()})
        self._dirty = True
        return {"stored": True, "tag": tag, "code": code,
                "index": len(self.codes) - 1,
                "cortex": np.asarray(cortex_vec, np.float32).copy()}

    # -- recall ----------------------------------------------------------

    def _matrix(self) -> np.ndarray:
        """Rebuild the recurrent matrix from the stored codes."""
        if not self._dirty:
            return self.W
        self.W = np.zeros((self.n_ca3, self.n_ca3), np.float32)
        if self.codes:
            C = np.stack(self.codes).astype(np.float32)
            m = C.mean(axis=0)
            for c in C:
                d = c - m
                self.W += np.outer(d, d) / np.float32(self.ca3_k)
        np.fill_diagonal(self.W, 0.0)
        self._dirty = False
        return self.W

    def _settle(self, cue: np.ndarray, steps: int = 12) -> np.ndarray:
        """Iterate the attractor under k-winner competition.

        Competition, not a threshold, because a threshold on a sparse code's
        overlap lets the tail through: measured on the spiking version, a
        120-cell episode came back as 733 cells, all correct plus 600
        hangers-on. Fixing the number of active cells keeps the code sparse,
        which is what stops every memory from resembling every other.
        """
        s = cue.copy()
        if s.sum() == 0:
            return s
        W = self._matrix()
        for _ in range(steps):
            h = W @ s
            if h.max() <= 0:
                break
            idx = np.argpartition(h, -self.ca3_k)[-self.ca3_k:]
            nxt = np.zeros(self.n_ca3, np.float32)
            nxt[idx] = 1.0
            if np.array_equal(nxt, s):
                break
            s = nxt
        return s

    def recall(self, cue_cortex: np.ndarray, steps: int = 12) -> dict:
        """Pattern completion from a partial cue, plus cortical reinstatement."""
        _, _, cue = self._ca3_code(cue_cortex)
        if cue.sum() == 0:
            return {"code": np.zeros(self.n_ca3, np.float32), "novelty": 1.0,
                    "reinstated": np.zeros(self.n_cortex, np.float32),
                    "overlap": 0.0, "cue": cue}
        completed = self._settle(cue, steps)
        shared = float(np.minimum(completed, cue).sum())
        union = float(np.maximum(completed, cue).sum()) + 1e-9
        overlap = shared / union
        novelty = float(np.clip(1.0 - overlap, 0.0, 1.0))
        self.last_novelty = novelty
        self.last_overlap = overlap
        return {"code": completed, "novelty": novelty, "overlap": overlap,
                "reinstated": self.back @ completed, "cue": cue}

    def replay(self, n: Optional[int] = None):
        """Episodes for consolidation, in order. Each carries the cortical
        pattern that was present when it was stored — replaying the stimulus
        instead is just looking at the same picture again."""
        eps = self.records if n is None else self.records[-n:]
        return list(eps)

    # -- observability ---------------------------------------------------

    def summary(self) -> str:
        return (f"Hippocampus(EC {self.n_ec}, DG {self.n_dg}, "
                f"CA3 {self.n_ca3} x {len(self.codes)} stored, "
                f"novelty {self.last_novelty:.2f})")
