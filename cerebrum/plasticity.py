"""Plasticity rules: spike-timing, reward modulation, sleep consolidation.

Every rule is a small object with one method — ``apply(synapses, pre, post,
dopamine)`` — so a brain's rule set is a list, not a hard-coded pipeline. Want
plain STDP? Pass ``[STDP()]``. Want reward-gated learning? Pass
``[STDP(), Dopamine()]``. Want nothing? Pass ``[]``.

What is deliberately *not* here: gradients. There is no loss and no backward
pass. Learning is local — a synapse changes using only the two neurons it
connects, plus a scalar that stands in for dopamine.
"""

from __future__ import annotations

from typing import Any, List, Optional

from . import backends


class STDP:
    """Pair-based spike-timing dependent plasticity.

    A synapse is potentiated when its presynaptic neuron fired recently and its
    postsynaptic neuron fires now (causal order), and depressed in the opposite
    order. Both terms read traces sampled *before* this step's spikes are
    folded in, so a spike cannot potentiate itself.

    Cost is proportional to ``spikes × fan_in``, never to the size of the
    matrix: at a 1% firing rate on a million neurons that is ~20k synapse
    updates per step.

    Parameters
    ----------
    ltp, ltd:
        Potentiation and depression rates. ``ltd`` slightly smaller than
        ``ltp`` keeps the network from collapsing to silence.
    tau_trace:
        Trace decay in milliseconds.
    clip:
        Hard bound on ``|w|``, applied every update.
    """

    def __init__(self, ltp: float = 0.006, ltd: float = 0.0045,
                 tau_trace: float = 20.0, clip: float = 4.0,
                 soft: bool = True) -> None:
        self.ltp = float(ltp)
        self.ltd = float(ltd)
        self.tau_trace = float(tau_trace)
        self.clip = float(clip)
        # Soft (multiplicative) bounds. With hard clipping every synapse is
        # driven to ±clip and the weight matrix becomes binary noise; the
        # measured effect on ten-pattern representation separation was
        # -0.11 against +0.05 for a non-plastic control. Scaling the update by
        # (1 - w/clip) makes strong synapses saturate smoothly and weak ones
        # catch up, which is the condition for competition to produce
        # selectivity instead of saturation.
        self.soft = bool(soft)
        self.name = "stdp"
        self._pre_trace: Any = None
        self._post_trace: Any = None
        self._backend_name = ""

    # -- per-brain state ------------------------------------------------

    def bind(self, brain) -> None:
        """Create trace arrays sized to this brain."""
        self._backend_name = brain.backend_name
        b = brain.backend
        self._pre_trace = b.zeros(brain.n_neurons)
        self._post_trace = b.zeros(brain.n_neurons)

    def spec(self) -> dict:
        return {"name": "stdp", "ltp": self.ltp, "ltd": self.ltd,
                "tau_trace": self.tau_trace, "clip": self.clip, "soft": self.soft}

    def state(self) -> dict:
        return {"pre_trace": self._pre_trace, "post_trace": self._post_trace}

    def load_state(self, data: dict) -> None:
        if "pre_trace" in data:
            self._pre_trace = data["pre_trace"]
        if "post_trace" in data:
            self._post_trace = data["post_trace"]

    # -- update ---------------------------------------------------------

    def apply(self, synapses, pre_spikes, post_spikes, dopamine: float = 1.0) -> int:
        """Update weights from the spikes of one step. Returns synapses touched."""
        if self._pre_trace is None:
            raise RuntimeError("STDP.bind(brain) must be called first")
        if dopamine == 0.0:
            return 0

        backend = backends.load(self._backend_name)
        decay = 1.0 - 1.0 / self.tau_trace
        pre_t = self._pre_trace
        post_t = self._post_trace
        touched = 0

        if backend.NAME == "numpy":
            import numpy as np

            w = synapses.w
            post = synapses.post
            pre = synapses.pre

            if len(pre_spikes):
                sids = synapses.gather_out(pre_spikes)
                if sids.size:
                    # depression: pre fired after its target did
                    dw = (-self.ltd * dopamine) * post_t[post[sids]]
                    if self.soft:
                        dw = dw * (1.0 + w[sids] / self.clip)
                    np.add.at(w, sids, dw)
                    touched += int(sids.size)
            if len(post_spikes):
                sids = synapses.gather_in(post_spikes)
                if sids.size:
                    dw = self.ltp * dopamine * pre_t[pre[sids]]
                    if self.soft:
                        dw = dw * (1.0 - w[sids] / self.clip)
                    np.add.at(w, sids, dw)
                    touched += int(sids.size)
            np.clip(w, -self.clip, self.clip, out=w)

            # decay, then fold in this step's spikes
            pre_t *= decay
            post_t *= decay
            if len(pre_spikes):
                pre_t[np.asarray(pre_spikes, np.int64)] += 1.0
            if len(post_spikes):
                post_t[np.asarray(post_spikes, np.int64)] += 1.0
            return touched

        # pure backend
        w = synapses.w
        post = synapses.post
        pre = synapses.pre
        if len(pre_spikes):
            for p in pre_spikes:
                for sid in synapses.synapses_from(p):
                    w[sid] -= self.ltd * dopamine * post_t[post[sid]]
                    touched += 1
        if len(post_spikes):
            for q in post_spikes:
                for sid in synapses.synapses_into(q):
                    w[sid] += self.ltp * dopamine * pre_t[pre[sid]]
                    touched += 1
        for i in range(len(w)):
            if w[i] > self.clip:
                w[i] = self.clip
            elif w[i] < -self.clip:
                w[i] = -self.clip
        for i in range(len(pre_t)):
            pre_t[i] *= decay
            post_t[i] *= decay
        for p in pre_spikes:
            pre_t[p] += 1.0
        for q in post_spikes:
            post_t[q] += 1.0
        return touched


class Dopamine:
    """Reward prediction error, as a scalar gate on plasticity.

    Holds a running baseline of received reward; the signal other rules see is
    ``reward - baseline``. This is the three-factor form: eligibility from
    spike timing (in :class:`STDP`), multiplied by a global neuromodulator.

    ``reward()`` is called by the outside world; ``value()`` is read by the
    brain each step.
    """

    def __init__(self, baseline_rate: float = 0.05, initial: float = 0.0) -> None:
        self.baseline_rate = float(baseline_rate)
        self.baseline = float(initial)
        self.last_reward = 0.0
        self._signal = 0.0
        self.name = "dopamine"

    def bind(self, brain) -> None:
        return None

    def reward(self, value: float) -> float:
        """Report a reward; returns the prediction error it produced."""
        value = float(value)
        self.last_reward = value
        error = value - self.baseline
        self.baseline += self.baseline_rate * error
        self._signal = error
        return error

    def value(self) -> float:
        """Modulator level as a gain on plasticity.

        ``1.0`` means "learn normally" — the resting state. Reward above
        baseline pushes it toward 2, below baseline toward 0, and it decays
        back to 1 so an old reward does not gate learning forever.
        """
        return max(0.0, 1.0 + self._signal)

    def tick(self, decay: float = 0.9) -> None:
        self._signal *= decay

    def spec(self) -> dict:
        return {"name": "dopamine", "baseline_rate": self.baseline_rate}

    def state(self) -> dict:
        return {"baseline": [self.baseline], "signal": [self._signal]}

    def load_state(self, data: dict) -> None:
        if "baseline" in data:
            self.baseline = float(list(data["baseline"])[0])
        if "signal" in data:
            self._signal = float(list(data["signal"])[0])


class Sleep:
    """Synaptic renormalisation and pruning — the consolidation pass.

    Runs offline, not per step. Two effects:

    1. **Renormalise** each neuron's incoming weight mass toward ``target``.
       Without this, STDP with net potentiation ratchets total drive upward
       until the population saturates; this is the brake.
    2. **Prune** synapses below ``prune_below``. Weak connections are noise;
       dropping them also shrinks the memory file.

    Both are measurable: total incoming mass per neuron is preserved to within
    float error, and the synapse count drops.
    """

    def __init__(self, target: float = 12.0, prune_below: float = 0.02) -> None:
        self.target = float(target)
        self.prune_below = float(prune_below)
        self.name = "sleep"

    def bind(self, brain) -> None:
        """No per-brain state; present so every rule has the same surface."""
        return None

    def spec(self) -> dict:
        return {"name": "sleep", "target": self.target,
                "prune_below": self.prune_below}

    def consolidate(self, synapses) -> dict:
        """One consolidation pass. Returns statistics."""
        backend = backends.load("numpy") if _is_numpy(synapses) else backends.load("pure")
        before = len(synapses)

        # Order matters: prune first, then renormalise. Normalising first and
        # pruning after leaves every neuron below its target mass (measured
        # 4.96 against a target of 5.0), because pruning removes weight the
        # normalisation had already accounted for.
        if backend.NAME == "numpy":
            import numpy as np

            w = synapses.w
            post = synapses.post
            synapses.compact(np.abs(w) >= self.prune_below)
            w = synapses.w
            post = synapses.post
            mass = np.bincount(post, weights=np.abs(w), minlength=synapses.n_post)
            scale = np.where(mass > 0, self.target / np.maximum(mass, 1e-9), 1.0)
            w *= scale[post].astype(w.dtype)
        else:
            keep = [abs(x) >= self.prune_below for x in synapses.w]
            _compact_pure(synapses, keep)
            mass = [0.0] * synapses.n_post
            for sid, q in enumerate(synapses.post):
                mass[q] += abs(synapses.w[sid])
            for sid, q in enumerate(synapses.post):
                if mass[q] > 0:
                    synapses.w[sid] *= self.target / mass[q]

        return {"before": before, "after": len(synapses),
                "pruned": before - len(synapses)}


def _is_numpy(synapses) -> bool:
    return hasattr(synapses, "_in_indptr")


def _compact_pure(synapses, keep) -> None:
    pre, post, w = [], [], []
    for i, k in enumerate(keep):
        if k:
            pre.append(synapses.pre[i])
            post.append(synapses.post[i])
            w.append(synapses.w[i])
    synapses.pre[:] = pre
    synapses.post[:] = post
    synapses.w[:] = w
    synapses._out = [[] for _ in range(synapses.n_pre)]
    for sid, p in enumerate(pre):
        synapses._out[p].append(sid)


class Pruning:
    """Structural plasticity: drop the weakest fraction of synapses.

    Separate from :class:`Sleep` because pruning and renormalising are
    different claims — one changes the graph, the other only rescales it.
    """

    def __init__(self, fraction: float = 0.02, floor: float = 0.0) -> None:
        self.fraction = float(fraction)
        self.floor = float(floor)
        self.name = "prune"

    def bind(self, brain) -> None:
        return None

    def spec(self) -> dict:
        return {"name": "prune", "fraction": self.fraction, "floor": self.floor}

    def apply(self, synapses) -> dict:
        before = len(synapses)
        if before == 0:
            return {"before": 0, "after": 0, "pruned": 0}
        if _is_numpy(synapses):
            import numpy as np

            w = np.abs(synapses.w)
            cut = np.quantile(w, self.fraction)
            keep = w > max(cut, self.floor)
            synapses.compact(keep)
        else:
            n = int(before * self.fraction)
            order = sorted(range(before), key=lambda i: abs(synapses.w[i]))
            drop = set(order[:n])
            _compact_pure(synapses, [i not in drop for i in range(before)])
        after = len(synapses)
        return {"before": before, "after": after, "pruned": before - after}


RULE_TYPES = {"stdp": STDP, "dopamine": Dopamine, "sleep": Sleep, "prune": Pruning}


def make_rule(spec: dict):
    """Rebuild a rule from its ``spec()`` dict."""
    spec = dict(spec)
    name = spec.pop("name")
    cls = RULE_TYPES.get(name)
    if cls is None:
        raise ValueError(f"unknown rule {name!r}")
    return cls(**spec)


class Structural:
    """Synapses grow and die on their own. No layout is imposed.

    This is the honest counterpart to hand-drawn regions: instead of a
    teacher deciding which neurons belong to "sensory" and which to
    "association", the graph changes by two local rules —

    * **grow**: when two neurons are active in the same window, a new synapse
      may appear between them (weight starts small, so only repeatedly
      co-active pairs survive the pruning that follows);
    * **prune**: the weakest synapses are removed on a schedule.

    Nothing about the resulting topology is specified in advance. The only
    things imposed from outside are the survival rules — growth needs
    co-activity, weights decay, and the circuit must keep firing in a band.
    That is the "teacher and doctor" role: stimulus and health, not
    architecture.

    Parameters
    ----------
    grow, prune:
        Fractions of the current synapse count to add and remove per pass.
    interval:
        Steps between structural passes.
    coactive_threshold:
        How much recent co-activity is needed before a candidate pair is
        allowed to connect.
    """

    def __init__(self, grow: float = 0.01, prune: float = 0.01,
                 interval: int = 200, coactive_threshold: float = 0.3) -> None:
        self.grow = float(grow)
        self.prune = float(prune)
        self.interval = int(interval)
        self.coactive_threshold = float(coactive_threshold)
        self.name = "structural"
        self._activity = None
        self._backend_name = ""
        self._steps = 0
        self.grown_total = 0
        self.pruned_total = 0

    def bind(self, brain) -> None:
        self._backend_name = brain.backend_name
        b = brain.backend
        self._activity = b.zeros(brain.n_neurons)

    def spec(self) -> dict:
        return {"name": "structural", "grow": self.grow, "prune": self.prune,
                "interval": self.interval,
                "coactive_threshold": self.coactive_threshold}

    # -- per-step bookkeeping --------------------------------------------

    def apply(self, synapses, pre_spikes, post_spikes, dopamine: float = 1.0) -> int:
        """Track activity; structural changes happen on the interval."""
        if self._activity is None:
            return 0
        if self._backend_name == "numpy":
            import numpy as np

            self._activity *= 0.995
            if len(post_spikes):
                self._activity[np.asarray(post_spikes, np.int64)] += 1.0
        else:
            for i in range(len(self._activity)):
                self._activity[i] *= 0.995
            for k in post_spikes:
                self._activity[k] += 1.0

        self._steps += 1
        if self.interval and self._steps % self.interval == 0:
            self.restructure(synapses)
        return 0

    # -- the structural pass ---------------------------------------------

    def restructure(self, synapses) -> dict:
        """Grow co-active pairs, prune the weakest. Returns the change."""
        before = len(synapses)
        if before == 0:
            return {"before": 0, "after": 0, "grown": 0, "pruned": 0}

        if self._backend_name == "numpy":
            import numpy as np

            act = self._activity
            live = np.nonzero(act > self.coactive_threshold)[0]
            n_grow = int(before * self.grow)
            if live.size > 1 and n_grow > 0:
                src = live[np.arange(n_grow) % live.size]
                dst = live[(np.arange(n_grow) * 7 + 3) % live.size]
                scale = 0.3 / max(1.0, len(live)) ** 0.5
                w = np.full(n_grow, scale, np.float32)
                synapses.add_batch(src.astype(np.int32),
                                   dst.astype(np.int32), w)

            n_prune = int(len(synapses) * self.prune)
            if n_prune > 0:
                w_abs = np.abs(synapses.w)
                cut = np.partition(w_abs, n_prune - 1)[n_prune - 1]
                synapses.compact(w_abs > cut)
        else:
            n_grow = int(before * self.grow)
            live = [i for i, a in enumerate(self._activity)
                    if a > self.coactive_threshold]
            if len(live) > 1:
                for k in range(n_grow):
                    src = live[k % len(live)]
                    dst = live[(k * 7 + 3) % len(live)]
                    synapses.add(src, dst, 0.05)
            n_prune = int(len(synapses) * self.prune)
            if n_prune > 0:
                order = sorted(range(len(synapses)),
                               key=lambda i: abs(synapses.w[i]))
                drop = set(order[:n_prune])
                _compact_pure(synapses,
                              [i not in drop for i in range(len(synapses))])

        after = len(synapses)
        grown = max(0, after - before + int(before * self.prune))
        self.grown_total += grown
        self.pruned_total += int(before * self.prune)
        return {"before": before, "after": after,
                "grown": grown, "pruned": before - after + grown}


STRUCTURAL = Structural


class Eligibility:
    """Three-factor learning: a synaptic tag, cashed in by dopamine.

    This is the biological answer to credit assignment, and it needs no
    gradient. Every synapse keeps a short-lived tag counting how much its
    presynaptic and postsynaptic cells fired together recently. When
    dopamine arrives — which can be a hundred milliseconds after the action
    that earned it — the tagged synapses are the ones that change:

        Δw = rate × (dopamine − resting level) × eligibility

    Nothing is backpropagated. A synapse only knows its own two neurons and
    one scalar that reaches every synapse in the brain.

    Parameters
    ----------
    decay:
        Per-step decay of the tag. At ~0.99 a tag survives about a hundred
        milliseconds, which is the window that lets a reward reach the
        synapses responsible for it.
    rate:
        How much weight one unit of dopamine adds per unit of tag.
    """

    def __init__(self, decay: float = 0.99, rate: float = 0.02,
                 clip: float = 4.0, interval: int = 20,
                 threshold: float = 0.05) -> None:
        self.decay = float(decay)
        self.rate = float(rate)
        self.clip = float(clip)
        # Cash tags in, and decay them, every ``interval`` steps rather than
        # every step. Decaying a 50M-element trace array cost about as much as
        # the whole simulation: a fly-scale run went from 17 ms/step to over
        # 40, and a 6-episode experiment took 26 minutes instead of 7. Dopamine
        # release is phasic, not per-millisecond, so batching the readout is
        # both cheaper and closer to the biology.
        self.interval = max(1, int(interval))
        self.threshold = float(threshold)
        self._acc = 0.0
        self._tick = 0
        self.name = "eligibility"
        self.E = None
        self._pre_trace = None
        self._backend_name = ""

    def bind(self, brain) -> None:
        self._backend_name = brain.backend_name
        b = brain.backend
        n = len(brain.recurrent)
        self.E = b.zeros(n)
        self._pre_trace = b.zeros(brain.n_neurons)

    def spec(self) -> dict:
        return {"name": "eligibility", "decay": self.decay,
                "rate": self.rate, "clip": self.clip}

    def apply(self, synapses, pre_spikes, post_spikes, dopamine: float = 1.0) -> int:
        """Tag co-active synapses, then cash in whatever dopamine is present."""
        if self.E is None:
            return 0
        import numpy as np

        E = self.E
        w = synapses.w
        E *= self.decay                     # tags fade unless refreshed

        touched = 0
        if len(post_spikes):
            sids = synapses.gather_in(post_spikes)
            if sids.size:
                # a synapse is tagged by (pre activity) × (post fired now)
                E[sids] += self._pre_trace[synapses.pre[sids]]
                touched = int(sids.size)

        # Batch the cash-in. ``dopamine - 1`` is accumulated between readouts
        # so a reward arriving mid-window is not thrown away.
        # Both the decay and the write touch the whole trace array, so at fly
        # scale (50M synapses) they cost ~50 ms each — more than the entire
        # simulation. Batching them every `interval` steps and skipping
        # insignificant rewards brings that to a few ms on average. Measured:
        # per-step readout ran at 110 ms/step and rewrote all 50M weights every
        # step; batched it is ~24 ms and only writes when something happened.
        self._acc += (dopamine - 1.0)
        self._tick += 1
        if self._tick % self.interval == 0:
            E *= self.decay ** self.interval
            if abs(self._acc) > self.threshold:
                w += (self.rate / self.interval) * self._acc * E
                np.clip(w, -self.clip, self.clip, out=w)
            self._acc = 0.0

        self._pre_trace *= 0.95
        if len(pre_spikes):
            self._pre_trace[np.asarray(pre_spikes, np.int64)] += 1.0
        return touched

    def reset_tags(self) -> None:
        """Called after a reward is cashed in: tags are spent."""
        if self.E is not None:
            self.E *= 0.0
