"""Thalamus: the relay and gate between the senses and the cortex.

Why it has to exist
-------------------
Without a thalamus the sensory vector is injected straight into the cortex,
which means *every* afferent is always equally important. A cortex wired that
way cannot separate signal from noise, and it has no way to be told "look at
this one" or "ignore that one".

The circuit implemented here is the classical one:

* **TC** (thalamocortical relay) — the projection cells. They carry the
  sensory value forward, but their gain is set by two inhibitory sources.
* **TRN** (thalamic reticular nucleus) — a shell of GABAergic cells that
  surrounds the relay. It is driven by *corticothalamic* feedback, so the
  cortex can suppress the channels it is not interested in. This is the
  biological implementation of attention: not amplifying the target, but
  damping the distractors.
* **GPi input** — the basal ganglia hold the relay under tonic inhibition and
  release it when an action is selected (see :mod:`cerebrum.brain.basal_ganglia`).
* **Post-inhibitory rebound** — a channel held down and then released fires a
  short high-frequency burst. That burst is a salience marker: it is how the
  cortex learns that something just became relevant.

The two modes are real: tonic (faithful, graded relay) and burst (novelty
signalling). This implementation keeps a single integrate-and-fire sheet but
reproduces both behaviours through the rebound term.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .sheet import NeuronSheet


class Thalamus:
    """Relay nuclei with reticular gating.

    Parameters
    ----------
    n_channels:
        Number of parallel sensory channels. One channel is one labelled line
        (a whisker, an odour, a colour-opponent channel).
    cells_per_channel:
        Relay cells per channel. More cells means a smoother rate estimate
        and a longer burst.
    sensory_gain:
        Multiplier on the incoming sensory vector.
    feedback_gain:
        Strength of the corticothalamic drive onto relay cells. Zero by
        default, and that is the physiologically correct value: the
        corticothalamic projection is a *modulator*, not a driver — it
        contacts the reticular nucleus far more strongly than the relay
        cells, so its net effect is suppression, not amplification. With the
        direct term non-zero the net effect inverted and attending a channel
        made it louder (measured: 4-10x increase), which is the opposite of
        what a thalamic gate is for.
    trn_gain:
        Strength of cortical drive onto the reticular nucleus (the "modulator"
        part). Raising this relative to ``feedback_gain`` turns cortical
        feedback into suppression, which is the attention mechanism.
    gpi_gain:
        Strength of the basal ganglia's tonic inhibition.
    """

    def __init__(
        self,
        n_channels: int = 16,
        cells_per_channel: int = 10,
        *,
        sensory_gain: float = 1.2,
        feedback_gain: float = 0.0,
        trn_gain: float = 0.6,
        gpi_gain: float = 1.8,
        rebound: float = 1.4,
        shunt: float = 40.0,
        seed: int = 0,
    ) -> None:
        self.n_ch = int(n_channels)
        self.k = int(cells_per_channel)
        n = self.n_ch * self.k

        self.sensory_gain = float(sensory_gain)
        self.feedback_gain = float(feedback_gain)
        self.trn_gain = float(trn_gain)
        self.gpi_gain = float(gpi_gain)
        self.rebound = float(rebound)
        self.shunt = float(shunt)

        # A 1-D strip: channel c owns cells [c*k, (c+1)*k). Positions follow
        # the index, so the distance kernel keeps channels from bleeding into
        # each other, and the strip's x coordinate is the channel coordinate
        # that the cortex maps topographically.
        self.tc = NeuronSheet(
            # A relay nucleus, not a cortex: very little recurrence, almost
            # no local inhibition. The gate is the reticular nucleus, not the
            # relay's own microcircuit, and making the relay recurrent turned
            # it into an oscillator that could not hold a steady rate.
            # No per-cell homeostasis on the relay. A relay cell's job is to
            # pass the signal through at whatever rate the drive sets; giving
            # it synaptic scaling made it fight its own input and then take
            # hundreds of steps to recover (measured: gain pinned at the
            # floor for 800 steps after a bout of firing, during which the
            # nucleus was silent).
            "thal_TC", n, shape=(1, n), excitatory_frac=0.98,
            p_local=0.02, sigma_conn=0.03, target_hz=12.0, w_norm=0.4,
            tau_m=20.0, eta_th=0.0, g_init=1.6, g_min=1.6, g_max=1.6,
            bias_mean=0.02, bias_std=0.02, th_max=2.5, g_adapt=0.3,
            seed=seed + 1,
        )
        # The reticular nucleus is a gate, not a self-regulating tissue. With
        # homeostasis on, its rate saturated near 5 Hz whatever the cortex
        # asked for, and the gate became a coin flip: a channel was closed in
        # only ~10% of milliseconds. Fixed gain makes its output track the
        # descending drive over the whole range, which is what a gate has to
        # do.
        self.trn = NeuronSheet(
            "thal_TRN", n, shape=(1, n), excitatory_frac=0.0,
            p_local=0.08, sigma_conn=0.05, target_hz=40.0, w_norm=0.4,
            bias_mean=0.05, bias_std=0.02, th_max=2.0,
            eta_th=0.0, g_init=1.6, g_min=1.6, g_max=1.6, seed=seed + 2,
        )
        self._n = n

        # TRN -> TC, same channel, dense within the block.
        self.trn_to_tc = np.zeros((n, n), np.float32)
        for c in range(self.n_ch):
            lo, hi = c * self.k, (c + 1) * self.k
            self.trn_to_tc[lo:hi, lo:hi] = -7.5 / max(1.0, self.k ** 0.5)

        self._suppressed = np.zeros(self.n_ch, np.float32)  # history, for rebound
        self._relay_ema = np.zeros(self.n_ch, np.float32)
        self.gate = np.ones(self.n_ch, np.float32)          # effective openness
        self.steps_total = 0
        self.spikes_total = 0

    # -- helpers ---------------------------------------------------------

    def channel_of(self, idx: np.ndarray) -> np.ndarray:
        return np.asarray(idx, np.int64) // self.k

    def channel_rate(self) -> np.ndarray:
        """Firing fraction per channel this step (0..1)."""
        out = np.zeros(self.n_ch, np.float32)
        if self.tc.prev.size:
            np.add.at(out, self.channel_of(self.tc.prev), 1.0)
        return out / max(1.0, self.k)

    # -- dynamics --------------------------------------------------------

    def step(
        self,
        sense: np.ndarray,
        cortical_feedback: Optional[np.ndarray] = None,
        gpi: Optional[np.ndarray] = None,
        *,
        relay_gain: float = 1.0,
    ) -> dict:
        """Advance 1 ms.

        Parameters
        ----------
        sense:
            Length ``n_channels``. Sensory intensity per channel.
        cortical_feedback:
            Length ``n_channels``. Descending cortical drive. It reaches the
            relay as a driver and the reticular nucleus as a modulator, so a
            strong feedback can end up net-suppressive for channels the
            cortex is not tracking.
        gpi:
            Length ``n_channels``. Basal ganglia output; high means held shut.

        Returns
        -------
        dict with ``relay`` (per-channel firing fraction), ``tc``/``trn``
        spike indices, ``gate`` (openness) and ``burst`` (per-channel flag).
        """
        sense = np.asarray(sense, np.float32).ravel()
        if sense.size != self.n_ch:
            raise ValueError(
                f"thalamus expects {self.n_ch} channels, got {sense.size}"
            )
        fb = (np.zeros(self.n_ch, np.float32) if cortical_feedback is None
              else np.asarray(cortical_feedback, np.float32).ravel())
        hold = (np.zeros(self.n_ch, np.float32) if gpi is None
                else np.asarray(gpi, np.float32).ravel())

        # External current, per channel, broadcast to that channel's cells.
        drive = sense * self.sensory_gain * relay_gain
        drive = drive + fb * self.feedback_gain

        # Rebound: a channel that was held down last step and is released now
        # gets a transient. This is the post-inhibitory burst; it is what
        # makes a *released* channel stand out from an *always-open* one.
        burst = np.clip(self._suppressed - 0.25, 0.0, 1.0) * self.rebound
        drive = drive + burst

        ext = np.repeat(drive - hold * self.gpi_gain, self.k).astype(np.float32)

        # TRN is driven by the same sensory stream and by cortex, but *not*
        # by GPi: the reticular nucleus is the cortex's instrument.
        trn_drive = np.repeat(
            (sense * 0.15 + fb * self.trn_gain).astype(np.float32), self.k)
        trn_idx = self.trn.step(trn_drive)

        # The reticular shunt is applied to the relay's *output*, not as
        # synaptic current into it. The reason is measured, not stylistic:
        # when the inhibition was delivered as current, the relay cells'
        # homeostatic gain compensated it away (their firing fell, their gain
        # climbed to its ceiling, and the next dip in inhibition produced a
        # burst), so driving the reticular nucleus made the attended channel
        # louder by a factor of eight. A shunt at the axon initial segment
        # divides the output and cannot be undone by synaptic scaling.
        tc_idx = self.tc.step(ext)

        rate = self.channel_rate()

        # Shunting inhibition is divisive, not subtractive. An inhibitory
        # synapse on the soma opens chloride channels and divides whatever
        # excitatory current arrives; it does not subtract a fixed amount.
        # Implementing it as subtraction is why the gate behaved backwards in
        # this model: the relay fired *more* when attended. Reading the
        # reticular rate out as a divisor reproduces the real effect — a
        # channel the cortex suppresses loses gain in proportion.
        trn_rate = np.zeros(self.n_ch, np.float32)
        if trn_idx.size:
            np.add.at(trn_rate, self.channel_of(trn_idx), 1.0)
        trn_rate /= max(1.0, self.k)
        gate = np.clip(1.0 - self.shunt * trn_rate, 0.05, 1.0)
        rate = rate * gate

        self._relay_ema += (rate - self._relay_ema) * 0.05
        self.gate = np.clip(1.0 - hold, 0.0, 1.0) * gate
        # track how hard each channel was pushed down, for next step's rebound
        # track how hard each channel was pushed down, for next step's rebound
        self._suppressed = np.clip(
            self._suppressed + (hold - self._suppressed) * 0.3, 0.0, 1.5)

        self.steps_total += 1
        self.spikes_total += int(tc_idx.size)

        return {
            "relay": rate,
            "relay_smooth": self._relay_ema.copy(),
            "tc": tc_idx,
            "trn": trn_idx,
            "gate": self.gate.copy(),
            "burst": (burst > 0.1).astype(np.float32),
        }

    def reset(self) -> None:
        self.tc.reset()
        self.trn.reset()
        self._suppressed[:] = 0.0

    def summary(self) -> str:
        return (f"Thalamus({self.n_ch} channels x {self.k} cells, "
                f"TC {self.tc.rate_hz():.1f} Hz, TRN {self.trn.rate_hz():.1f} Hz)")
