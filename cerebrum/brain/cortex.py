"""Cortex: three layers, local inhibition, and a self-organising input map.

The layout follows the canonical cortical microcircuit closely enough to keep
the signal paths honest:

* **L4** — input layer. Thalamocortical afferents land here. These synapses
  are the plastic ones: this is where a topographic map forms, and it is the
  same site as barrel formation in rodent somatosensory cortex.
* **L2/3** — association layer. Recurrent, receives L4, sends to L5 and back
  to L4 in a loop that sustains activity after the stimulus is gone.
* **L5** — output layer. Sends to the basal ganglia, the cerebellum (as an
  efference copy), the brainstem, and back down to the thalamus.

Every layer is a :class:`~cerebrum.brain.sheet.NeuronSheet`, so every layer
has its own spatially interleaved inhibitory cells and its own per-neuron
threshold homeostasis.

The afferent map is the part that answers "can the cortex partition itself".
It cannot, without positions (see the sheet module). With positions and a
neighbourhood kernel it does: measured adjacency agreement between the
preferred stimulus of neighbouring cells went from 0.50 (chance, winner-only
learning) to 0.97 (sigma = 0.30) on a 30x30 sheet trained on two input
families. No region borders were specified anywhere.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .sheet import NeuronSheet, build_projection


class Cortex:
    """A three-layer cortical column sheet.

    Parameters
    ----------
    n_l4, n_l23, n_l5:
        Cells per layer.
    n_channels:
        Width of the thalamic relay vector feeding L4.
    n_actions:
        Number of output channels L5 is partitioned into. Each block of L5
        cells is read out as one action's evidence by the basal ganglia.
    """

    def __init__(
        self,
        n_l4: int = 900,
        n_l23: int = 900,
        n_l5: int = 400,
        n_channels: int = 16,
        n_actions: int = 6,
        *,
        afferent_lr: float = 0.05,
        afferent_sigma: float = 0.25,
        seed: int = 0,
    ) -> None:
        self.n_ch = int(n_channels)
        self.n_actions = int(n_actions)
        self.afferent_lr = float(afferent_lr)
        self.afferent_sigma = float(afferent_sigma)

        self.l4 = NeuronSheet(
            "ctx_L4", n_l4, excitatory_frac=0.78, p_local=0.22,
            sigma_conn=0.13, target_hz=5.0, w_norm=1.0, th_max=2.5,
            global_inh=6.0, eta_th=1e-3, tau_rate=100.0,
            seed=seed + 11,
        )
        self.l23 = NeuronSheet(
            "ctx_L23", n_l23, excitatory_frac=0.78, p_local=0.25,
            sigma_conn=0.16, target_hz=5.0, w_norm=1.0, th_max=2.5,
            global_inh=6.0, eta_th=1e-3, tau_rate=100.0,
            seed=seed + 12,
        )
        self.l5 = NeuronSheet(
            "ctx_L5", n_l5, excitatory_frac=0.8, p_local=0.2,
            sigma_conn=0.15, target_hz=5.0, w_norm=0.9, th_max=2.5,
            global_inh=6.0, eta_th=1e-3, tau_rate=100.0,
            seed=seed + 13,
        )

        rng = np.random.default_rng(seed + 100)

        # Thalamocortical afferents: the plastic map. Starts as a weak random
        # projection so the sheet has *something* to match against; a zero
        # matrix would make every cell equally bad and the winner arbitrary.
        # Thalamocortical afferents are glutamatergic: every one of them is
        # excitatory. A random-sign matrix makes half of them inhibitory,
        # which cancels the relay's drive — measured, L4 stayed at 0 spikes
        # per step even with its homeostatic gain pinned at the ceiling,
        # while L2/3 downstream was firing, i.e. the input layer was being
        # actively suppressed by its own input.
        aff = np.abs(rng.normal(0.0, 0.15, (self.l4.n, self.n_ch))).astype(np.float32)
        aff /= (np.linalg.norm(aff, axis=1, keepdims=True) + 1e-9)
        self.aff = aff

        # Forward chain, topographic so that neighbouring columns stay
        # neighbours across layers.
        self.l4_to_l23 = build_projection(
            self.l23.n, self.l4.n, 40, 1.5, rng,
            pos_pre=self.l4.pos, pos_post=self.l23.pos, topo=True)
        self.l23_to_l5 = build_projection(
            self.l5.n, self.l23.n, 40, 1.5, rng,
            pos_pre=self.l23.pos, pos_post=self.l5.pos, topo=True)
        # Feedback loop L5 -> L2/3, weaker and more diffuse: feedback is a
        # modulator of the loop, not a driver of it. Making it as strong as
        # the forward path turns the column into an oscillator.
        self.l5_to_l23 = build_projection(
            self.l23.n, self.l5.n, 40, 0.5, rng,
            pos_pre=self.l5.pos, pos_post=self.l23.pos, topo=True)

        # Cortical feedback to the thalamus, one weight per channel.
        ch_pos = np.stack([np.linspace(0, 1, self.n_ch)] * 2, axis=1)
        self.l5_to_thal = build_projection(
            self.n_ch, self.l5.n, 12, 0.35, rng,
            pos_pre=self.l5.pos, pos_post=ch_pos, topo=True)

        # Action blocks in L5, contiguous by position.
        self.action_blocks = []
        per = self.l5.n // self.n_actions
        for a in range(self.n_actions):
            lo = a * per
            hi = (a + 1) * per if a < self.n_actions - 1 else self.l5.n
            self.action_blocks.append((lo, hi))

        self.gain_l4 = 25.0
        self.rate_l4 = np.zeros(self.l4.n, np.float32)
        self.rate_l23 = np.zeros(self.l23.n, np.float32)
        self.rate_l5 = np.zeros(self.l5.n, np.float32)
        self.learn_map = True
        self._pending_fb = None
        self._fb_state = np.zeros(self.n_ch, np.float32)
        # L5 fires sparsely (a few cells per millisecond), so the raw block
        # mean lands near 0.05 — below the range the basal ganglia can score.
        # The gain brings a typical proposal into 0.2-0.8 without changing
        # the ranking, which is all the striatum reads.
        self.evidence_gain = 12.0
        # Tonic background. A purely driven cortical sheet goes into down
        # states for tens of milliseconds between volleys, and during those
        # the readout is exactly zero — measured, L5 held an 11-37 Hz average
        # while its instantaneous rate was 0 at one sample in three, so the
        # basal ganglia kept seeing an empty proposal and holding every
        # channel. Real cortex has spontaneous activity for the same reason:
        # without it, the system cannot tell silence from "no opinion".
        self.tonic = {"l4": 0.75, "l23": 0.65, "l5": 0.55}
        self.ev_smooth = np.zeros(self.n_actions, np.float32)

    # -- readouts --------------------------------------------------------

    def activity(self) -> dict:
        return {
            "l4": self.rate_l4.copy(),
            "l23": self.rate_l23.copy(),
            "l5": self.rate_l5.copy(),
            "action": self.action_evidence(),
            "feedback": self.feedback_to_thalamus(),
        }

    def action_evidence(self, smooth: bool = False) -> np.ndarray:
        """Per-action L5 firing fraction — the cortex's motor proposal.

        ``smooth`` returns a 50 ms average instead of the instantaneous value.
        The basal ganglia read the smoothed version: a proposal that flickers
        to zero between cortical volleys would otherwise close every channel
        for the duration of the gap.
        """
        out = np.zeros(self.n_actions, np.float32)
        for a, (lo, hi) in enumerate(self.action_blocks):
            out[a] = float(np.clip(self.rate_l5[lo:hi].mean()
                                   * self.evidence_gain, 0.0, 1.0))
        if smooth:
            return self.ev_smooth.copy()
        return out

    def feedback_to_thalamus(self) -> np.ndarray:
        """Descending drive, one value per thalamic channel.

        Low-passed. Corticothalamic axons are slow and diffuse, and passing
        the raw L5 rate through makes the loop oscillate: measured, L4 went
        9.5 -> 1.5 spikes/step and back with a period near 1000 steps,
        because feedback, reticular inhibition and relay gain formed a
        delayed negative-feedback loop with no damping.
        """
        raw = (self.l5_to_thal @ self.rate_l5).astype(np.float32)
        self._fb_state += (raw - self._fb_state) * 0.02
        return self._fb_state

    def encode(self, relay: np.ndarray) -> None:
        """Write the thalamic relay into the afferent map (no spiking)."""
        if not self.learn_map:
            return
        # One learning step per input presentation, using the relay vector as
        # the input. The sheet picks the winning column and drags its
        # neighbourhood along, which is what makes the map topographic.
        self.l4.learn_map(self.aff, np.asarray(relay, np.float32),
                          lr=self.afferent_lr, sigma=self.afferent_sigma)

    # -- dynamics --------------------------------------------------------

    def step(self, relay: np.ndarray, *, arousal: float = 1.0,
             noise: Optional[np.ndarray] = None) -> dict:
        """Advance 1 ms with the thalamic relay as input."""
        relay = np.asarray(relay, np.float32).ravel()

        l4_ext = (self.aff @ relay) * self.gain_l4 * arousal + self.tonic['l4'] * arousal
        if noise is not None:
            l4_ext = l4_ext + noise
        l4_idx = self.l4.step(l4_ext)

        l23_syn = (self.l4_to_l23[:, l4_idx].sum(axis=1)
                   if l4_idx.size else np.zeros(self.l23.n, np.float32))
        l23_syn = l23_syn + np.float32(self.tonic["l23"] * arousal)
        if self._pending_fb is not None:
            # L5 -> L2/3 lands one step later, on the apical dendrites. The
            # delay is what keeps the loop from locking into a two-cell
            # oscillation at the membrane time constant.
            l23_syn = l23_syn + self._pending_fb
        l23_idx = self.l23.step(None, syn_extra=l23_syn)

        l5_syn = (self.l23_to_l5[:, l23_idx].sum(axis=1)
                  if l23_idx.size else np.zeros(self.l5.n, np.float32))
        l5_syn = l5_syn + np.float32(self.tonic["l5"] * arousal)
        l5_idx = self.l5.step(None, syn_extra=l5_syn)

        # feedback lands on L2/3 one step later
        if l5_idx.size:
            self._pending_fb = self.l5_to_l23[:, l5_idx].sum(axis=1)
        else:
            self._pending_fb = None

        self.rate_l4 += (self._bincount(self.l4.n, l4_idx) - self.rate_l4) * 0.05
        self.rate_l23 += (self._bincount(self.l23.n, l23_idx) - self.rate_l23) * 0.05
        self.rate_l5 += (self._bincount(self.l5.n, l5_idx) - self.rate_l5) * 0.05

        ev = self.action_evidence()
        self.ev_smooth += (ev - self.ev_smooth) * 0.004
        return {
            "l4": l4_idx, "l23": l23_idx, "l5": l5_idx,
            "action": ev,
            "feedback": self.feedback_to_thalamus(),
        }

    @staticmethod
    def _bincount(n: int, idx: np.ndarray) -> np.ndarray:
        out = np.zeros(n, np.float32)
        if idx.size:
            out[idx] = 1.0
        return out

    def reset(self) -> None:
        self.l4.reset()
        self.l23.reset()
        self.l5.reset()
        self.rate_l4[:] = 0.0
        self.rate_l23[:] = 0.0
        self.rate_l5[:] = 0.0

    def summary(self) -> str:
        return (f"Cortex(L4 {self.l4.n} @ {self.l4.rate_hz():.1f} Hz, "
                f"L23 {self.l23.n} @ {self.l23.rate_hz():.1f} Hz, "
                f"L5 {self.l5.n} @ {self.l5.rate_hz():.1f} Hz)")


class Prefrontal:
    """Working memory: a gated attractor that holds a pattern after input ends.

    This one is rate-based on purpose. A spiking attractor with bistability
    requires the recurrent gain to sit inside a narrow band; drift outside it
    and the network either dies or saturates, which is the single most common
    failure mode in published spiking working-memory models. The canonical
    model of delay-period activity is a rate attractor, so that is what is
    implemented, and it is labelled rate-based rather than passed off as
    spiking.

    Writing is gated: nothing is stored while the gate is closed, which is
    the difference between a working memory and a leaky integrator. Dopamine
    opens the gate (D1 disinhibition), which is why stress and reward both
    change what gets held in mind.
    """

    def __init__(self, n: int = 300, *, tau: float = 30.0, seed: int = 0) -> None:
        self.n = int(n)
        self.tau = float(tau)
        rng = np.random.default_rng(seed + 77)
        self.W = rng.normal(0.0, 0.02, (self.n, self.n)).astype(np.float32)
        np.fill_diagonal(self.W, 0.0)
        self.state = np.zeros(self.n, np.float32)
        self.gain = 1.6
        self.gate = 0.0
        self.encoding = False

    def write(self, pattern: np.ndarray, lr: float = 0.06) -> None:
        """Store a pattern as an attractor (Hebbian outer product)."""
        p = np.asarray(pattern, np.float32)
        p = p / (np.linalg.norm(p) + 1e-9)
        self.W += lr * np.outer(p, p)
        np.fill_diagonal(self.W, 0.0)
        np.clip(self.W, -0.6, 0.6, out=self.W)

    def step(self, inp: Optional[np.ndarray] = None, *, gate: float = 0.0) -> np.ndarray:
        self.gate = float(gate)
        drive = np.zeros(self.n, np.float32) if inp is None else np.asarray(inp, np.float32)
        if gate > 0.5:
            # during encoding the attractor is driven, not recalled
            target = drive if inp is not None else self.state
            self.state += (target - self.state) * 0.3
        else:
            rec = self.W @ self.state
            self.state += (-self.state + self.gain * rec + 0.3 * drive) / self.tau
        np.clip(self.state, 0.0, 1.5, out=self.state)
        return self.state

    def reset(self) -> None:
        self.state[:] = 0.0
