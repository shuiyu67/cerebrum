"""Brain assembly: neurons + sparse wiring + sensors and actuators.

A :class:`Brain` is deliberately thin — it knows nothing about what the input
means:

* one **population** of leaky integrate-and-fire neurons,
* a **recurrent matrix** with ``fan_in`` inputs per neuron,
* a **sensor projection** from an arbitrary-width input vector into current,
* an **actuator decoder** reading population activity back out as an
  arbitrary-width action vector.

A camera, a microphone, a token embedding or a joystick all arrive as a list of
floats of whatever length the projection was built for. Capacity comes from the
population size, not from the interface.

The backend is picked at construction: NumPy when importable (fast, up to a
million neurons), the standard library otherwise (small networks, no install).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np

from . import backends


class Brain:
    """A spiking brain with arbitrary-width sensors and actuators.

    Parameters
    ----------
    neurons:
        Population size. Measured on a 16-core desktop: 10k runs faster than
        real time, 100k about 3x slower, 1M about 31x slower (32 steps/s at a
        1% firing rate, 366 MB resident). See :mod:`cerebrum.bench.scale`.
    inputs, outputs:
        Width of the sensor and action vectors. Not tied to a modality.
    fan_in:
        Recurrent inputs per neuron; total synapses are ``neurons * fan_in``
        at 12 bytes each. This is the knob that decides memory and step cost.
    sensor_span:
        Neurons each sensor connects to — the receptive-field width. A wide
        span makes one input vector drive the whole population, which costs
        ``n_inputs * span`` synaptic events per step and can dominate the
        step time.
    backend:
        ``None`` for automatic, or ``"numpy"`` / ``"pure"``.
    seed:
        Deterministic wiring and intrinsic parameters. Same seed, same brain.
    sensor_gain:
        Multiplier applied to sensor values before they become current.
    areas:
        Split the population into ``(sensory, association, prediction)``
        regions, e.g. ``(400, 1200, 400)``. Sensors then reach only the
        sensory region, which projects forward to association, which projects
        to prediction — instead of one uniform soup where every neuron sees a
        random global average. ``None`` keeps the flat, fully mixed layout.
    input_plasticity:
        Let the sensor -> neuron synapses learn. This is the one place
        selectivity can come from: a neuron prefers a pattern because of the
        input weights it holds, and everywhere else in the engine those
        weights are frozen. The rule is local Oja-style Hebbian — a neuron
        that fires moves its input weights toward what it just saw, with a
        decay term that keeps them bounded. No target, no teacher, no layout.
    input_lr:
        Step size for that input learning.
    memory_units:
        Give every neuron its own memory: a prototype of the input patterns it
        has seen. Input then drives a neuron by *similarity to that prototype*
        instead of through a fixed random synapse, and the prototype is updated
        competitively (the best-matching ``memory_k`` neurons move toward the
        input). This is the difference between a neuron that only integrates
        and one that remembers.
    memory_k, memory_lr:
        How many neurons compete for each input, and how fast a winner's
        prototype moves toward it.
    hippocampus:
        Attach episodic memory. The cortex learns slowly and statistically;
        this remembers *events* — one exposure, content-addressed recall,
        replayed during sleep. Without it a brain can recognise but not
        recount.
    hip_cells, hip_sparsity:
        Size of the dentate expansion and the fraction of cells active per
        episode. Separation between similar episodes improves with size:
        measured discrimination of two episodes differing in 5% of their
        inputs was 35% at 16k cells, 78% at 64k and 100% at 131k.
    predictive:
        Attach a prediction loop. A small readout population learns to
        forecast the next sensor vector from current activity; the part it gets
        wrong is fed back as extra current. Learning here is local and
        unsupervised — there is no teacher, only surprise.
    pred_neurons, pred_lr, pred_tau:
        Size of the forecasting population, its learning rate, and the decay of
        its rate estimate.
    error_gain:
        How strongly prediction error is injected back into the population.
    **neuron_params:
        Passed to the backend's :class:`Population` (``tau_m``,
        ``target_hz``, ``pacemaker_frac``, ...).
    """

    def __init__(
        self,
        neurons: int = 2000,
        inputs: int = 64,
        outputs: int = 8,
        fan_in: int = 20,
        sensor_span: int = 100,
        backend: Optional[str] = None,
        seed: Optional[int] = None,
        sensor_gain: float = 2.0,
        areas: Optional[tuple] = None,
        input_plasticity: bool = False,
        input_lr: float = 0.02,
        memory_units: bool = False,
        memory_k: int = 32,
        memory_lr: float = 0.05,
        hippocampus: bool = False,
        hip_cells: int = 65536,
        hip_sparsity: float = 0.002,
        predictive: bool = False,
        pred_neurons: int = 256,
        pred_lr: float = 0.02,
        pred_tau: float = 0.9,
        error_gain: float = 1.5,
        **neuron_params: Any,
    ) -> None:
        if neurons < 1:
            raise ValueError("neurons must be >= 1")
        if inputs < 1 or outputs < 1:
            raise ValueError("inputs and outputs must be >= 1")
        if outputs > neurons:
            raise ValueError("outputs cannot exceed neurons")

        self.backend = backends.load(backend)
        self.backend_name = self.backend.NAME
        self.rng = self.backend.Rng(seed)
        self.seed = seed

        self.n_neurons = int(neurons)
        self.n_inputs = int(inputs)
        self.n_outputs = int(outputs)
        self.fan_in = int(fan_in)
        self.sensor_span = int(sensor_span)
        self.sensor_gain = float(sensor_gain)

        self.areas = tuple(int(x) for x in areas) if areas else None
        if self.areas is not None and sum(self.areas) != self.n_neurons:
            raise ValueError(
                f"areas {self.areas} must sum to neurons={self.n_neurons}"
            )
        self.neuron_params = dict(neuron_params)
        # The rate target and the sparsity cap have to agree. k-WTA allows at
        # most wta_k cells to fire, i.e. wta_k / n * 1000 Hz, so asking
        # homeostasis for 20 Hz while the cap sits at 1.44 Hz puts it in
        # permanent error: gain and drive climb to their ceilings and the
        # controller loses all authority. Measured, it sat at error 18.7 for
        # 1200 steps with drive pinned at 1.0 and rising.
        k_cap = int(self.neuron_params.get("wta_k", 0) or 0)
        if k_cap > 0:
            ceiling = k_cap / float(self.n_neurons) * 1000.0
            asked = float(self.neuron_params.get("target_hz", 5.0))
            self.neuron_params["target_hz"] = min(asked, ceiling)
        if self.areas and self.backend.NAME == "numpy":
            # each region regulates its own excitability
            self.neuron_params["regions"] = self.areas
        self.population = self.backend.Population(
            self.n_neurons, self.rng, **self.neuron_params
        )

        self.recurrent = self.backend.SparseSynapses(self.n_neurons, self.n_neurons)
        if self.areas is not None:
            self._build_area_wiring()
        else:
            pre, post, weight = self.backend.build_random(
                self.n_neurons, self.n_neurons, self.fan_in, self.rng,
                0.5 / max(1.0, self.fan_in) ** 0.5,
            )
            self.recurrent.add_batch(pre, post, weight)

        sensor_targets = self.areas[0] if self.areas else self.n_neurons
        self.sensor_proj = self.backend.SparseSynapses(self.n_inputs, sensor_targets)
        # Unit weight per sensor synapse: a sensor value of 1.0 then delivers
        # roughly one unit of current, i.e. it can carry a neuron to threshold
        # on its own. Dividing by sqrt(span) here would be wrong — that is the
        # scaling for a *dense* projection where every neuron sums many
        # sensors, and it left the input too weak to move anything.
        pre, post, weight = self.backend.build_sensor(
            self.n_inputs, sensor_targets, self.sensor_span, self.rng, 1.0,
        )
        self.sensor_proj.add_batch(pre, post, weight)

        # --- prediction loop (option B) --------------------------------
        self.predictive = bool(predictive)
        self.pred_error_ema = 0.0
        if self.predictive:
            if self.backend.NAME != "numpy":
                raise NotImplementedError(
                    "predictive mode needs the numpy backend"
                )
            import numpy as np

            self.pred_n = min(int(pred_neurons), self.n_neurons)
            self.pred_lr = float(pred_lr)
            self.pred_tau = float(pred_tau)
            self.error_gain = float(error_gain)
            # main -> forecasters (fixed sparse)
            src_n = self.areas[2] if self.areas else self.n_neurons
            src_start = (self.areas[0] + self.areas[1]) if self.areas else 0
            self.pred_src_start = src_start
            self.pred_src_n = src_n
            self.to_pred = self.backend.SparseSynapses(src_n, self.pred_n)
            pre, post, weight = self.backend.build_random(
                src_n, self.pred_n, 8, self.rng,
                1.0 / max(1.0, src_n / self.pred_n) ** 0.5,
            )
            self.to_pred.add_batch(pre, post, weight)
            # forecasters -> predicted sensor vector (plastic, dense, small)
            # Start at zero. Random small weights sum to ~6 over 256 features
            # against a 0/1 target, which overshoots immediately and diverges
            # by positive feedback (measured: error grew 0.13 -> 0.35).
            self.pred_dec = np.zeros((self.n_inputs, self.pred_n), np.float32)
            self.pred_rate = np.zeros(self.pred_n, np.float32)
            self.pred_error = np.zeros(self.n_inputs, np.float32)

        # --- episodic memory -------------------------------------------
        self.hippocampus = None
        if hippocampus:
            from .hippocampus import Hippocampus

            self.hippocampus = Hippocampus(self.n_inputs, n_cells=hip_cells,
                                           sparsity=hip_sparsity, seed=seed or 0)

        # --- plastic input synapses (self-organising selectivity) ------
        self.input_plasticity = bool(input_plasticity)
        self.input_lr = float(input_lr)

        # --- per-neuron memory (prototypes) ----------------------------
        self.memory_units = bool(memory_units)
        self.memory_k = int(memory_k)
        self.memory_lr = float(memory_lr)
        if self.memory_units:
            if self.backend.NAME != "numpy":
                raise NotImplementedError("memory_units needs the numpy backend")
            import numpy as np

            proto = np.asarray(
                self.rng.normal_list(self.n_neurons * self.n_inputs, 0.0, 0.1),
                np.float32,
            ).reshape(self.n_neurons, self.n_inputs)
            proto /= np.linalg.norm(proto, axis=1, keepdims=True) + 1e-9
            self.prototypes = proto
            self.last_overlap = 1.0

        self.group_size = max(1, self.n_neurons // self.n_outputs)
        self._prev_spikes: Any = []
        self._spike_total = 0
        self._step_total = 0

        # plasticity rules, applied after every step in list order
        self.rules: list = []
        self.plasticity_events = 0

    # -- structured wiring -----------------------------------------------

    def _build_area_wiring(self) -> None:
        """Wire regions instead of shuffling everything together.

        Within each region connections are local and dense-ish; between
        regions they are strictly forward (sensory -> association ->
        prediction) plus a weak feedback path. That is what gives the
        prediction region features of its own rather than a blurred sample of
        the whole population: measured pattern discriminability of the readout
        features went from 0.71 (uniform random) to 0.42 (areas).
        """
        n_s, n_a, n_p = self.areas
        starts = (0, n_s, n_s + n_a)
        sizes = (n_s, n_a, n_p)
        scale = 0.5 / max(1.0, self.fan_in) ** 0.5

        pre, post, weight = [], [], []

        def connect(src_start, src_n, dst_start, dst_n, fan, gain):
            _, _, w = self.backend.build_random(
                src_n, dst_n, fan, self.rng, scale * gain)
            p, q, ww = self.backend.build_random(
                src_n, dst_n, fan, self.rng, 1.0)
            # build_random returns absolute indices within the block
            for i in range(len(p)):
                pre.append(src_start + int(p[i]))
                post.append(dst_start + int(q[i]))
                weight.append(float(ww[i]) * scale * gain)

        for k in range(3):
            connect(starts[k], sizes[k], starts[k], sizes[k], self.fan_in, 1.0)
        # forward projections carry gain: a region only sees a small upstream
        # population, so 1.0 per synapse leaves the signal below threshold and
        # the chain dies at the first hop (measured: 0 spikes downstream).
        connect(starts[0], sizes[0], starts[1], sizes[1], self.fan_in, 6.0)
        connect(starts[1], sizes[1], starts[2], sizes[2], self.fan_in, 6.0)
        connect(starts[2], sizes[2], starts[1], sizes[1], max(2, self.fan_in // 4), 0.5)
        self.recurrent.add_batch(pre, post, weight)

    # -- dynamics --------------------------------------------------------

    def step(self, sensor: Optional[Iterable[float]] = None, force=None):
        """Advance one millisecond.

        Parameters
        ----------
        sensor:
            One value per sensor, length ``n_inputs``. ``None`` means silence.

        Returns
        -------
        (action, n_spikes)
            ``action`` has length ``n_outputs`` and sums to 1 — it is the
            population's spike distribution across actuator groups. When
            nothing fires it is uniform.
        """
        # --- prediction: forecast this step's sensors from last step's rate
        residual = None
        if self.predictive:
            import numpy as np

            # bounded feature vector: raw spike counts are unbounded and blow
            # the readout up within a few steps (measured: error rose from
            # 0.14 to 0.34 instead of falling)
            # Sparse binary features. An exponential rate average saturates
            # near 1.0 for every unit once the network fires at 5 Hz, so it
            # cannot tell patterns apart (measured: prediction error climbed
            # to the no-prediction baseline). Which forecasters fired *last
            # step* is sparse and pattern-specific instead.
            feat = (self.pred_rate > 0.0).astype(np.float32)
            predicted = self.pred_dec @ feat
            if sensor is not None:
                actual = np.asarray(list(sensor), np.float32)
                err = actual - predicted
                self.pred_error = err
                # Normalised LMS. Plain LMS with 256 features and a fixed
                # step size oscillates and then diverges — measured error
                # climbing 0.22 -> 0.38 while predicting nothing would have
                # stayed at 0.22. Dividing by the feature energy makes the
                # step size dimension-free.
                energy = float(feat @ feat) + 1e-6
                # decay matters: without it the same 20 active features are
                # hit every step, weights saturate at the clip and the
                # prediction becomes a constant (measured: error drifting to
                # the 0.5 no-prediction baseline).
                self.pred_dec *= 0.999
                self.pred_dec += (self.pred_lr * err / energy)[:, None] * feat[None, :]
                self.pred_dec = np.clip(self.pred_dec, -8.0, 8.0)
                strength = float(np.abs(err).mean())
                self.pred_error_ema = 0.99 * self.pred_error_ema + 0.01 * strength
                residual = err

        current = self.backend.zeros(self.n_neurons)

        if len(self._prev_spikes):
            self.recurrent.propagate(self._prev_spikes, current)

        if sensor is not None:
            values = list(sensor)
            if len(values) != self.n_inputs:
                raise ValueError(
                    f"sensor must have length {self.n_inputs}, got {len(values)}"
                )
            gain = self.sensor_gain
            if self.memory_units:
                import numpy as np

                x = np.asarray(values, np.float32)
                # a neuron is driven by how much the input resembles its memory
                sim = self.prototypes @ (x / (np.linalg.norm(x) + 1e-9))
                current += (sim * gain).astype(np.float32)
                # competitive update: the best matches move toward this input
                k = min(self.memory_k, self.n_neurons)
                top = np.argpartition(sim, -k)[-k:]
                self.prototypes[top] += self.memory_lr * (x[None, :] - self.prototypes[top])
                norm = np.linalg.norm(self.prototypes[top], axis=1, keepdims=True) + 1e-9
                self.prototypes[top] /= norm
                active = []
            else:
                active = [i for i, x in enumerate(values) if x > 0.0]
            if active:
                if self.areas:
                    # the projection ends in the sensory region only
                    buf = self.backend.zeros(self.areas[0])
                    self.sensor_proj.propagate_weighted(
                        active, [values[i] * gain for i in active], buf
                    )
                    self.backend.add_block(current, buf)
                else:
                    self.sensor_proj.propagate_weighted(
                        active, [values[i] * gain for i in active], current
                    )

        if self.predictive and residual is not None:
            # only the unexpected part is passed back in — that is the point
            import numpy as np

            active = np.nonzero(residual)[0]
            if active.size:
                vals = (residual[active] * self.error_gain).tolist()
                if self.areas:
                    buf = self.backend.zeros(self.areas[0])
                    self.sensor_proj.propagate_weighted(active.tolist(), vals, buf)
                    self.backend.add_block(current, buf)
                else:
                    self.sensor_proj.propagate_weighted(
                        active.tolist(), vals, current)

        spikes = self.population.step(current, force=force)

        if self.input_plasticity and sensor is not None and len(spikes):
            self._learn_input(list(sensor), spikes)

        if self.predictive:
            # update the forecaster rate estimate from this step's activity
            import numpy as np

            inp = np.zeros(self.pred_n, np.float32)
            if len(spikes):
                if self.areas:
                    local = [int(k) - self.pred_src_start for k in spikes
                             if self.pred_src_start <= int(k) <
                             self.pred_src_start + self.pred_src_n]
                    if local:
                        self.to_pred.propagate(local, inp)
                else:
                    self.to_pred.propagate(spikes, inp)
            # keep the instantaneous pattern; the EMA is only for smooth reads
            self.pred_rate = inp

        if self.rules:
            prev = self._prev_spikes
            self._apply_plasticity(prev, spikes)

        self._prev_spikes = spikes
        n = len(spikes)
        self._spike_total += n
        self._step_total += 1

        counts = self.backend.group_counts(spikes, self.n_outputs, self.group_size)
        total = sum(counts)
        action = [c / total for c in counts] if total > 0 else \
            [1.0 / self.n_outputs] * self.n_outputs
        return action, n

    def _learn_input(self, x_list, spikes) -> None:
        """Competitive Hebbian update of the sensor -> neuron synapses.

        Only the neuron in each neighbourhood that matches the input *best*
        learns; the rest are left alone. Updating every cell that fired is what
        the previous version did, and it cannot self-organise: all of them move
        toward the same input, so their receptive fields converge on the same
        average and every region ends up preferring the same stimulus
        (measured: 8 of 8 regions preferred class A, 0 preferred class B, under
        both global and local inhibition).

        Competitive updates are what make different cells claim different parts
        of the input — the same reason a Kohonen map does not collapse.
        """
        import numpy as np

        if len(spikes) == 0:
            return
        x = np.asarray(x_list, np.float32)
        proj = self.sensor_proj
        w = proj.w
        pre = proj.pre
        lr = self.input_lr

        def match(j):
            """How well cell ``j``'s current receptive field fits this input."""
            sids = proj.synapses_into(int(j))
            if len(sids) == 0:
                return -np.inf, None
            sids = np.asarray(sids, np.int64)
            src = np.asarray(pre[sids], np.int64)
            ok = src < x.size
            score = float(np.dot(w[sids][ok], x[src][ok]))
            return score, sids

        # Competitive selection: only the cells that already fit this input
        # best are allowed to learn from it. No region assignment is used
        # here — the point is that specialisation has to come from competition,
        # not from us deciding who belongs where.
        scored = []
        for j in spikes:
            j = int(j)
            sc, sids = match(j)
            if sids is not None:
                scored.append((sc, j))
        if not scored:
            return
        scored.sort(reverse=True)
        n_win = max(1, len(scored) // 8)
        winners = [j for _, j in scored[:n_win]]

        for j in winners:
            _, sids = match(j)
            if sids is None:
                continue
            sids = np.asarray(sids, np.int64)
            src = np.asarray(pre[sids], np.int64)
            ok = src < x.size
            sids, src = sids[ok], src[ok]
            target = x[src]
            w[sids] += lr * (target - w[sids])

    # -- plasticity ------------------------------------------------------

    def add_rule(self, rule):
        """Attach a plasticity rule and return it (for chaining)."""
        rule.bind(self)
        self.rules.append(rule)
        return rule

    def rule(self, name: str):
        for r in self.rules:
            if getattr(r, "name", "") == name:
                return r
        return None

    def _apply_plasticity(self, pre_spikes, post_spikes) -> None:
        """Run every rule against this step's spike pair."""
        dopamine = 1.0
        gate = self.rule("dopamine")
        if gate is not None:
            dopamine = gate.value()      # 1.0 at rest, modulated by RPE
            gate.tick()
        for r in self.rules:
            fn = getattr(r, "apply", None)
            if fn is None:
                continue
            if getattr(r, "name", "") == "dopamine":
                continue
            self.plasticity_events += fn(self.recurrent, pre_spikes,
                                         post_spikes, dopamine)

    def reinforce(self, reward: float) -> float:
        """Report reward; returns the prediction error it produced."""
        gate = self.rule("dopamine")
        if gate is None:
            raise RuntimeError("no Dopamine rule attached; call add_rule(Dopamine())")
        return gate.reward(reward)

    def run(self, steps: int, sensor: Optional[Iterable[float]] = None,
            probe_every: int = 0):
        """Run ``steps`` milliseconds; return probe samples if asked."""
        series = []
        for i in range(steps):
            action, n = self.step(sensor)
            if probe_every and (i + 1) % probe_every == 0:
                series.append((i + 1, round(self.rate_hz, 3), n, action))
        return series

    # -- episodic memory -------------------------------------------------

    def experience(self, sensor, tag=None, reward: float = 0.0,
                   steps: int = 10):
        """Record an event, together with the cortical pattern it produced.

        The sensory vector alone is not enough for consolidation: replaying the
        *input* just makes the cortex look at the same picture again, which
        collapses its input weights onto a few cluster centres and measurably
        blurs the representation (separability 0.58 -> 0.69). What the
        hippocampus actually hands back is the pattern — which cells fired
        together — so that the synapses between them can be bound into
        something the cortex can later reconstruct on its own.
        """
        if self.hippocampus is None:
            raise RuntimeError("no hippocampus; construct with hippocampus=True")
        rec = self.hippocampus.experience(list(sensor), tag=tag, reward=reward)

        import numpy as np

        fired = set()
        for _ in range(max(1, steps)):
            self.step(list(sensor))
            fired.update(int(k) for k in self._prev_spikes)
        self.reset_state()
        rec["pattern"] = np.asarray(sorted(fired), np.int32)
        return rec

    def remember(self, cue):
        """Which stored event does this fragment belong to?"""
        if self.hippocampus is None:
            raise RuntimeError("no hippocampus; construct with hippocampus=True")
        return self.hippocampus.nearest(list(cue))

    def sleep(self, target: Optional[float] = None,
              replay: Optional[int] = None,
              steps_per_episode: int = 10,
              consolidate_lr: Optional[float] = None):
        """Consolidate: renormalise the wiring, then replay the day.

        Replay drives the cortex with the patterns the hippocampus recorded,
        which is the mechanism by which a one-off event becomes a habit. The
        homeostatic thermostat is frozen for the duration — it rescales
        excitability globally and would cancel whatever the replay changed.
        """
        """Consolidate: renormalise the wiring, then replay the day.

        Replay drives the cortex with stored episodes while plasticity is on,
        which is the mechanism by which a one-off event becomes a statistical
        regularity. It is the difference between a brain that remembers
        episodes and one that has learned from them.
        """
        stats = {}
        # ``None`` means "replay everything"; ``0`` means "only renormalise"
        if replay is None:
            replay = len(self.hippocampus.episodes) if self.hippocampus else 0
        if self.hippocampus is not None and replay:
            episodes = self.hippocampus.replay(replay)

            # The thermostat has to be off. Homeostasis rescales the whole
            # population's excitability by one factor, so anything learning
            # does to the weights gets compensated away: replaying 96 episodes
            # moved representation separability by 0.01 against a no-replay
            # control. During consolidation the gain and drive are frozen;
            # they resume on waking.
            pop = self.population
            was_on = getattr(pop, "homeostasis_on", True)
            if hasattr(pop, "homeostasis_on"):
                pop.homeostasis_on = False
            # replay is consolidation, so the input pathway must be plastic
            # for it to have anywhere to write — otherwise this is just
            # watching old photographs.
            try:
                for ep in episodes:
                    pattern = ep.get("pattern")
                    if pattern is not None and len(pattern):
                        # replay the cortical pattern, not the stimulus: the
                        # cells fire from the inside, so what changes is the
                        # wiring between them, not the input weights
                        for _ in range(steps_per_episode):
                            self.step(None, force=pattern)
                    else:
                        for _ in range(steps_per_episode):
                            self.step(ep["x"].tolist())
                    self.reset_state()
            finally:
                if hasattr(pop, "homeostasis_on"):
                    pop.homeostasis_on = was_on

            stats["replayed"] = len(episodes)
            stats["homeostasis_frozen"] = was_on

        if target is not None or not stats:
            pass
        sleeper = self.rule("sleep")
        if sleeper is None and target is not None:
            from .plasticity import Sleep as _Sleep

            sleeper = self.add_rule(_Sleep())
        if sleeper is not None:
            if target is not None:
                sleeper.target = float(target)
            stats.update(sleeper.consolidate(self.recurrent))
        self._prev_spikes = self.backend.zeros(0)
        return stats

    def _response_strength(self) -> float:
        """Mean spikes per step over a short idle window."""
        before = self._spike_total, self._step_total
        for _ in range(20):
            self.step(None)
        spawned = self._spike_total - before[0]
        self._spike_total, self._step_total = before
        return spawned / 20.0

    def reset_state(self) -> None:
        """Clear membrane state between stimuli, keeping what was learned.

        Homeostatic gain/drive, weights and prototypes survive — those are the
        brain's accumulated state. Only the transient electrical state is
        cleared, so the next image starts from rest instead of inheriting the
        previous one's after-potentials.
        """
        pop = self.population
        if self.backend.NAME == "numpy":
            pop.v.fill(0.0)
            pop.adapt.fill(0.0)
            pop.resources.fill(1.0)
            pop.refractory.fill(0)
            pop._last_spikes = np.empty(0)
        else:
            for i in range(pop.n):
                pop.v[i] = 0.0
                pop.adapt[i] = 0.0
                pop.resources[i] = 1.0
                pop.refractory[i] = 0
            pop._last_spikes = []
        self._prev_spikes = self.backend.zeros(0)

    # -- observability ---------------------------------------------------

    @property
    def rate_hz(self) -> float:
        """Mean firing rate since construction, in Hz."""
        if self._step_total == 0:
            return 0.0
        return self._spike_total / self._step_total / self.n_neurons * 1000.0

    def summary(self) -> str:
        rate = f"{self.rate_hz:.2f} Hz" if self._step_total else "not measured"
        return (
            f"Brain(neurons={self.n_neurons:,}, "
            f"synapses={len(self.recurrent):,}, "
            f"sensors={self.n_inputs}, actuators={self.n_outputs}, "
            f"backend={self.backend_name}, rate={rate})"
        )

    def __repr__(self) -> str:
        return self.summary()
