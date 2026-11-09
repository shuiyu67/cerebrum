"""The assembled brain.

This is the only module that knows how the pieces fit together. Every part
below it is unaware of the others, which is deliberate: it means a lesion
experiment can remove one and leave the rest working, and it means a claim
about a subsystem can be tested without the whole assembly in the loop.

Signal flow, one millisecond:

1. **Neuromodulators** decay and are read: cortical gain, learning rate,
   patience, arousal.
2. **Hypothalamus** updates the drives (hunger, fatigue, fear, orexin) from
   the body's state and the amygdala's output.
3. **Thalamus** receives the senses, the cortical feedback, and the basal
   ganglia's tonic inhibition. It relays what is allowed through.
4. **Cortex** runs L4 -> L2/3 -> L5 with the relay as input; the afferent map
   in L4 learns one step per presentation.
5. **Basal ganglia** score the cortical proposals and release one channel.
6. **Amygdala** receives the same senses by a direct route (fast, crude) and
   the cortical state by a slower one; if a shock arrives it tags the
   associated cue in a single trial.
7. **Cerebellum** gets an efference copy of the cortical command, predicts
   the error, and returns a timing correction.
8. **Hippocampus** silently accumulates episodes; every so often it is asked
   to complete the current situation from an incomplete cue, and the
   mismatch becomes the novelty signal that drives noradrenaline and
   acetylcholine.

The action is the cortex's proposal weighted by how far the basal ganglia
opened each channel, plus the hypothalamic urge. That last term is what makes
the animal act when nothing in the world is asking it to.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .amygdala import Amygdala
from .basal_ganglia import BasalGanglia
from .cerebellum import Cerebellum
from .cortex import Cortex, Prefrontal
from .hippocampus import Hippocampus
from .hypothalamus import Hypothalamus
from .neuromodulation import Neuromodulators
from .thalamus import Thalamus

def relay_proxy(vec: np.ndarray, n_channels: int) -> np.ndarray:
    """Compress a cortical population vector into a thalamic relay vector.

    Reinstatement has to re-enter the cortex through the same channel it left
    by, so the stored population pattern is projected onto the channel space
    the thalamus speaks.
    """
    v = np.asarray(vec, np.float32).ravel()
    if v.size == 0:
        return np.zeros(n_channels, np.float32)
    chunks = np.array_split(v, n_channels)
    return np.array([float(c.mean()) for c in chunks], np.float32)


SYSTEMS = ("thalamus", "cortex", "basal_ganglia", "hippocampus",
           "amygdala", "hypothalamus", "cerebellum", "neuromodulators")


class WholeBrain:
    """All of the above, wired into one loop.

    Parameters
    ----------
    n_sense, n_actions:
        Sensor and action widths.
    lesion:
        Names from :data:`SYSTEMS` to disable. A lesioned system is not
        removed from the object graph — it is bypassed, so the rest of the
        brain keeps running with the same shapes and the same seeds. That is
        what makes an ablation a controlled comparison rather than a
        different brain.
    """

    def __init__(self, n_sense: int = 16, n_actions: int = 6, *,
                 seed: int = 0, lesion: tuple = (),
                 n_cortex: int = 900) -> None:
        self.n_sense = int(n_sense)
        self.n_actions = int(n_actions)
        self.seed = int(seed)
        self.lesion = set(lesion)
        for name in self.lesion:
            if name not in SYSTEMS:
                raise ValueError(f"unknown system {name!r}; known: {SYSTEMS}")

        self.thalamus = Thalamus(n_sense, 10, seed=seed)
        self.cortex = Cortex(n_l4=n_cortex, n_l23=n_cortex, n_l5=400,
                             n_channels=n_sense, n_actions=n_actions, seed=seed)
        self.bg = BasalGanglia(n_actions, seed=seed)
        self.hippocampus = Hippocampus(n_cortex=400, seed=seed)
        self.amygdala = Amygdala(n_sense, seed=seed)
        self.hypothalamus = Hypothalamus(seed=seed)
        self.cerebellum = Cerebellum(n_mossy=n_sense + n_actions, seed=seed)
        self.modulators = Neuromodulators(seed=seed)
        self.prefrontal = Prefrontal(300, seed=seed)

        self.time_steps = 0
        self.last_action = -1
        self.last_relay = np.zeros(n_sense, np.float32)
        self.last_gpi = np.full(n_actions, 0.7)
        self.last_reward = 0.0
        self.last_error = 0.0
        self.episodes_stored = 0
        self._novelty = 1.0
        self.trace: list = []

    # -- one millisecond -------------------------------------------------

    def step(self, sensors: np.ndarray, *, us: float = 0.0,
             learn: bool = True) -> dict:
        """Advance the whole brain by 1 ms; returns a full state dict."""
        self.time_steps += 1
        sensors = np.asarray(sensors, np.float32).ravel()

        # 1. modulators tick first — they set the gains everything else uses
        if "neuromodulators" in self.lesion:
            # A brain with fixed neuromodulation: no phasic dopamine, so
            # nothing can be reinforced, and no arousal coupling.
            mods = {"da": 0.0, "da_tonic": 0.0, "serotonin": 0.5,
                    "ne": 0.3, "ach": 0.4, "orexin": 0.7, "rpe": 0.0}
            arousal = 1.0
        else:
            mods = self.modulators.step(uncertainty=self._novelty)
            arousal = self.modulators.cortical_gain()

        # 2. drives
        if "hypothalamus" in self.lesion:
            hypo = {"hunger": 0.5, "fatigue": 0.0, "fear": 0.0,
                    "orexin": 0.7, "crf": 0.0, "forage_bias": 0.0,
                    "rest_bias": 0.0, "arousal": 1.0}
        else:
            self.hypothalamus.observe(sensors)
            hypo = self.hypothalamus.step(
                fear_in=0.0 if "amygdala" in self.lesion
                else self.amygdala.fear)

        # 3. thalamic gate
        if "thalamus" in self.lesion:
            relay = sensors.copy()
            thal = {"relay": relay, "gate": np.ones(self.n_sense, np.float32),
                    "burst": np.zeros(self.n_sense, np.float32),
                    "tc": np.empty(0), "trn": np.empty(0)}
        else:
            fb = self.cortex.feedback_to_thalamus()
            gpi = (np.zeros(self.n_actions, np.float32)
                   if "basal_ganglia" in self.lesion else self.last_gpi)
            # map action channels onto sensory channels: each action channel
            # inhibits the thalamic channels it competes for. This is the
            # vl/va projection, crudely.
            hold = np.zeros(self.n_sense, np.float32)
            for a in range(self.n_actions):
                lo = (a * self.n_sense) // self.n_actions
                hi = ((a + 1) * self.n_sense) // self.n_actions
                hold[lo:hi] = max(hold[lo:hi].max() if hi > lo else 0.0, gpi[a])
            thal = self.thalamus.step(sensors, fb, hold)
            relay = 0.85 * self.last_relay + 0.15 * thal["relay"]

        self.last_relay = relay

        # 4. cortex
        ctx = self.cortex.step(relay, arousal=arousal)
        if learn:
            self.cortex.encode(relay)
        evidence = self.cortex.ev_smooth.copy()

        # 5. basal ganglia
        if "basal_ganglia" in self.lesion:
            gpi_out = np.zeros(self.n_actions, np.float32)
        else:
            gpi_out = self.bg.step(
                evidence, dopamine=self.modulators.da_tonic,
                hold=float(np.clip(hypo["crf"], 0.0, 1.0)))
        self.last_gpi = gpi_out

        # 6. amygdala: CS is the raw sense vector, US is external
        if "amygdala" in self.lesion:
            amyg = {"fear": 0.0, "prediction": 0.0,
                    "la": np.empty(0), "ce": np.empty(0)}
        else:
            amyg = self.amygdala.step(sensors, us, arousal=arousal)
            if learn and us > 0.0:
                self.amygdala.learn(sensors, us, arousal=arousal,
                                    ach=self.modulators.ach)
            if "hypothalamus" not in self.lesion:
                self.hypothalamus.fear = max(self.hypothalamus.fear, amyg["fear"])

        # 7. cerebellum: efference copy of what the cortex just proposed
        if "cerebellum" in self.lesion:
            cb = {"output": 0.0, "purkinje": np.zeros(1), "deep": np.empty(0),
                  "granule": np.empty(0)}
        else:
            mossy = np.concatenate([relay, evidence]).astype(np.float32)
            cb = self.cerebellum.step(mossy, error=self.last_error)

        # 8. action
        action, score = self._select(evidence, gpi_out, hypo, cb)

        # 9. consolidation and episodic storage happen on a slower clock
        if learn and "hippocampus" not in self.lesion:
            if self.time_steps % 250 == 0:
                self.hippocampus.experience(self.cortex.rate_l5.copy(),
                                            tag=f"t{self.time_steps}")
                self.episodes_stored += 1
            if self.time_steps % 500 == 0:
                rec = self.hippocampus.recall(self.cortex.rate_l5.copy(),
                                              steps=10)
                self._novelty = float(rec["novelty"])
                self.modulators.surprise(0.4 * self._novelty)
        else:
            self._novelty = 0.0

        self.last_action = int(action)
        out = {
            "action": int(action),
            "score": score,
            "evidence": evidence,
            "gpi": gpi_out,
            "release": 1.0 - gpi_out,
            "relay": relay,
            "fear": amyg["fear"],
            "hypo": hypo,
            "cb": cb["output"],
            "mods": mods,
            "novelty": self._novelty,
            "rates": {"l4": self.cortex.l4.rate_hz(),
                      "l23": self.cortex.l23.rate_hz(),
                      "l5": self.cortex.l5.rate_hz()},
            "gate": thal["gate"],
        }
        self.trace.append(out)
        if len(self.trace) > 4000:
            self.trace.pop(0)
        return out

    # -- choice ----------------------------------------------------------

    def _select(self, evidence, gpi, hypo, cb) -> tuple:
        """Cortex proposes, basal ganglia permits, hypothalamus insists."""
        ev = np.asarray(evidence, np.float64)
        release = 1.0 - np.asarray(gpi, np.float64)
        score = ev * (0.15 + 0.85 * release)

        bias = np.zeros(self.n_actions)
        # drives are additive urges on specific channels, not a replacement
        # for the cortical proposal — an animal with a strong urge but no
        # stimulus still has to pick *something*, and the cortex supplies it
        # Scaled so that a cortical proposal of ~0.3 can compete with a
        # drive of ~0.5. When the biases were larger the animal rested
        # regardless of what the cortex proposed and starved: measured, 1043
        # of the last 2000 steps were spent resting while hungry.
        bias[0] += 0.42 * max(0.0, hypo["forage_bias"])   # forward / explore
        bias[3] += 0.55 * max(0.0, hypo["forage_bias"])   # eat
        bias[5] += 0.30 * max(0.0, hypo["rest_bias"])     # rest
        bias[4] += 0.95 * float(hypo["fear"])             # flee
        score = score + bias

        # Cerebellar correction sharpens the timing of the leading channel
        # rather than adding an action of its own.
        if cb["output"] > 0:
            score[0] += 0.08 * float(np.tanh(cb["output"] / 20.0))

        score = score + np.random.default_rng(
            self.time_steps).normal(0.0, 0.01, self.n_actions)
        return int(np.argmax(score)), score

    # -- learning --------------------------------------------------------

    def reinforce(self, reward: float) -> float:
        """Report an outcome. Returns the prediction error."""
        if "neuromodulators" in self.lesion:
            self.last_reward = float(reward)
            self.last_error = float(np.clip(-reward, 0.0, 1.0))
            return 0.0
        rpe = self.modulators.reward(reward)
        self.last_reward = float(reward)
        self.last_error = float(np.clip(-reward, 0.0, 1.0))
        if "basal_ganglia" not in self.lesion:
            self.bg.learn(self.cortex.action_evidence(), self.last_action, rpe)
        # dopamine also gates the prefrontal write
        self.prefrontal.step(self.cortex.rate_l5[:self.prefrontal.n],
                             gate=1.0 if self.modulators.da_tonic > 0 else 0.0)
        return rpe

    def sleep(self, cycles: int = 3) -> dict:
        """Consolidate: replay episodes through the cortex.

        Replay drives the afferent map with the cortical patterns the
        hippocampus stored, which is the only route by which a single
        experience becomes a statistical regularity.
        """
        if "hippocampus" in self.lesion:
            return {"replayed": 0}
        eps = self.hippocampus.replay(cycles * 8)
        was = self.cortex.learn_map
        self.cortex.learn_map = True
        n = 0
        for ep in eps:
            vec = np.asarray(ep.get("cortex", np.zeros(0)), np.float32)
            if vec.size == 0:
                continue
            # reinstatement: drive the cortex with the pattern the episode
            # left behind, then let the afferent map absorb it
            proxy = relay_proxy(vec, self.n_sense)
            for _ in range(4):
                self.cortex.step(proxy)
            self.cortex.encode(proxy)
            n += 1
        self.cortex.learn_map = was
        self.reset_state()
        return {"replayed": n}

    def reset_state(self) -> None:
        """Clear transient electrical state across every spiking region."""
        self.cortex.reset()
        self.thalamus.reset()
        self.amygdala.reset()
        self.cerebellum.reset()

    # -- observability ---------------------------------------------------

    def report(self) -> dict:
        return {
            "steps": self.time_steps,
            "action": self.last_action,
            "gpi": self.last_gpi.copy(),
            "rates": {"l4": self.cortex.l4.rate_hz(),
                      "l23": self.cortex.l23.rate_hz(),
                      "l5": self.cortex.l5.rate_hz(),
                      "thal": self.thalamus.tc.rate_hz()},
            "mods": self.modulators.levels(),
            "hypo": {"hunger": self.hypothalamus.hunger,
                     "fatigue": self.hypothalamus.fatigue,
                     "fear": self.hypothalamus.fear,
                     "orexin": self.hypothalamus.orexin},
            "fear": self.amygdala.fear,
            "episodes": self.episodes_stored,
            "novelty": self._novelty,
        }

    def summary(self) -> str:
        return (f"WholeBrain({sum(s.n for s in
                                  (self.cortex.l4, self.cortex.l23, self.cortex.l5)):,} "
                f"cortical cells + thalamus/amygdala/cerebellum, "
                f"{self.hippocampus.ca3.n_stored} episodes, "
                f"t={self.time_steps})")
