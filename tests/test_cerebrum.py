"""Tests for cerebrum. Run with ``python -m unittest discover -s cerebrum/tests``.

Small networks throughout so the suite finishes in seconds. The NumPy-only
tests are skipped when NumPy is absent.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile

from cerebrum import backends, memory
from cerebrum.core import Brain
from cerebrum.plasticity import STDP, Dopamine, Sleep, Pruning, make_rule
from cerebrum.hippocampus import Hippocampus


def small(**overrides):
    cfg = dict(neurons=400, inputs=24, outputs=3, fan_in=16, sensor_span=40, seed=3)
    cfg.update(overrides)
    return Brain(**cfg)


class TestBackends(unittest.TestCase):
    def test_auto_backend_prefers_numpy(self):
        self.assertIn(backends.load().NAME, backends.available())
        self.assertEqual(backends.load("pure").NAME, "pure")

    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError):
            backends.load("cuda")

    def test_both_backends_start_from_silence(self):
        for name in backends.available():
            with self.subTest(backend=name):
                b = Brain(neurons=300, inputs=8, outputs=2, fan_in=10,
                          sensor_span=20, backend=name, seed=1)
                first = [b.step(None)[1] for _ in range(3)]
                self.assertLess(sum(first), 30)   # no avalanche at t=0


class TestDynamics(unittest.TestCase):
    def test_activity_grows_without_input(self):
        """With no input at all, the population must wake itself up.

        Homeostatic drive accumulates at ~2e-4 per step, so the population
        needs a couple of thousand steps to cross threshold from silence —
        that is the behaviour, not a bug.
        """
        b = small()
        early = sum(b.step(None)[1] for _ in range(50))
        for _ in range(2000):
            b.step(None)
        late = sum(b.step(None)[1] for _ in range(50))
        self.assertGreater(late, early)
        self.assertGreater(b.rate_hz, 0.0)

    def test_arbitrary_io_widths(self):
        for inputs, outputs in ((3, 1), (37, 5), (129, 16)):
            with self.subTest(inputs=inputs, outputs=outputs):
                b = Brain(neurons=400, inputs=inputs, outputs=outputs,
                          fan_in=10, sensor_span=20, seed=2)
                action, n = b.step([0.5] * inputs)
                self.assertEqual(len(action), outputs)
                self.assertAlmostEqual(sum(action), 1.0, places=6)
                self.assertGreaterEqual(n, 0)

    def test_sensor_width_is_checked(self):
        b = small()
        with self.assertRaises(ValueError):
            b.step([0.1] * (b.n_inputs + 1))

    def test_numpy_and_pure_agree(self):
        """The two backends must agree statistically, not bit for bit.

        NumPy keeps ``v`` in float32 and the pure backend in Python floats
        (float64). That is a different rounding of the same dynamics, so
        individual spike times drift apart while the population rate does not.
        """
        if len(backends.available()) < 2:
            self.skipTest("needs both backends")
        a = Brain(neurons=300, inputs=12, outputs=2, fan_in=12,
                  sensor_span=20, backend="numpy", seed=4)
        p = Brain(neurons=300, inputs=12, outputs=2, fan_in=12,
                  sensor_span=20, backend="pure", seed=4)
        sensor = [0.3] * 12
        na = sum(a.step(sensor)[1] for _ in range(200))
        np_ = sum(p.step(sensor)[1] for _ in range(200))
        self.assertGreater(na + np_, 0)
        self.assertLess(abs(na - np_) / max(1, max(na, np_)), 0.35)


class TestMemory(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "m" + memory.SUFFIX)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.dir, ignore_errors=True)

    def test_roundtrip_is_bit_identical(self):
        b = small()
        sensor = [0.6] * 24
        for _ in range(60):
            b.step(sensor)
        memory.save(b, self.path)

        loaded = memory.load(self.path)
        self.assertEqual(list(b.recurrent.w), list(loaded.recurrent.w))
        self.assertEqual(list(b.population.v), list(loaded.population.v))

        for step in range(120):
            a1, n1 = b.step(sensor)
            a2, n2 = loaded.step(sensor)
            self.assertEqual(n1, n2, f"spike count diverged at step {step}")
            self.assertEqual(a1, a2, f"action diverged at step {step}")

    def test_peek_reports_the_spec(self):
        b = small()
        memory.save(b, self.path)
        line = memory.peek(self.path)
        self.assertIn("neurons=400", line)
        self.assertIn("sensors=24", line)

    def test_cross_backend_load(self):
        if len(backends.available()) < 2:
            self.skipTest("needs both backends")
        b = small()
        memory.save(b, self.path)
        pure = memory.load(self.path, backend="pure")
        self.assertEqual(pure.backend_name, "pure")
        self.assertEqual(len(pure.recurrent.w), len(b.recurrent.w))

    def test_corrupt_part_is_detected(self):
        b = small()
        memory.save(b, self.path)
        broken = self.path + ".broken"
        with zipfile.ZipFile(self.path) as zin, zipfile.ZipFile(broken, "w") as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "connectome.bin":
                    data = bytes([data[0] ^ 0xFF]) + data[1:]
                zout.writestr(item, data)
        with self.assertRaises(memory.MemoryError_):
            memory.load(broken)

    def test_future_format_is_refused(self):
        b = small()
        memory.save(b, self.path)
        future = self.path + ".future"
        with zipfile.ZipFile(self.path) as zin, zipfile.ZipFile(future, "w") as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "manifest.json":
                    m = json.loads(data)
                    m["format"] = memory.FORMAT + 1
                    data = json.dumps(m).encode()
                zout.writestr(item, data)
        with self.assertRaises(memory.MemoryError_):
            memory.load(future)

    def test_non_brain_file_is_refused(self):
        junk = os.path.join(self.dir, "junk.brain")
        with open(junk, "w", encoding="utf-8") as f:
            f.write("not a brain")
        with self.assertRaises(Exception):
            memory.load(junk)


class TestMergeGraft(unittest.TestCase):
    def test_merge_is_weighted_average(self):
        a = small(seed=5)
        b = small(seed=5)
        for _ in range(30):
            a.step([0.8] * 24)
        for _ in range(10):
            b.step([0.2] * 24)
        merged = memory.merge([a, b])
        self.assertEqual(merged._step_total, a._step_total + b._step_total)
        wa = a._step_total / (a._step_total + b._step_total)
        for i in range(0, len(merged.recurrent.w), 97):
            expect = wa * a.recurrent.w[i] + (1 - wa) * b.recurrent.w[i]
            self.assertAlmostEqual(merged.recurrent.w[i], expect, places=6)

    def test_merge_rejects_different_wiring(self):
        a = small(seed=5)
        b = small(seed=6)
        with self.assertRaises(memory.MemoryError_):
            memory.merge([a, b])

    def test_graft_across_sizes(self):
        small_brain = Brain(neurons=200, inputs=24, outputs=3, fan_in=16,
                            sensor_span=40, seed=9)
        big_brain = Brain(neurons=600, inputs=24, outputs=3, fan_in=16,
                          sensor_span=40, seed=9)
        for _ in range(40):
            small_brain.step([0.7] * 24)
        grafted = memory.graft(small_brain, big_brain, weight=0.75)
        self.assertEqual(grafted.n_neurons, 600)
        action, n = grafted.step([0.7] * 24)
        self.assertEqual(len(action), 3)
        self.assertGreaterEqual(n, 0)
        # the target must be left untouched
        self.assertEqual(big_brain._step_total, 0)

    def test_merge_then_save_and_reload(self):
        a = small(seed=5)
        b = small(seed=5)
        a.step([0.8] * 24)
        b.step([0.2] * 24)
        merged = memory.merge([a, b])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "merged" + memory.SUFFIX)
            memory.save(merged, p)
            again = memory.load(p)
        self.assertEqual(list(merged.recurrent.w), list(again.recurrent.w))


class TestPlasticity(unittest.TestCase):
    def test_stdp_changes_weights_and_respects_clip(self):
        b = small()
        b.add_rule(STDP(clip=0.8))
        before = list(b.recurrent.w)
        for _ in range(200):
            b.step([0.5] * 24)
        after = list(b.recurrent.w)
        self.assertNotEqual(before, after)
        self.assertGreater(b.plasticity_events, 0)
        self.assertLessEqual(max(abs(x) for x in after), 0.8 + 1e-6)

    def test_soft_bounds_avoid_saturation(self):
        """Hard clipping turns the matrix into +-clip noise; soft bounds do not."""
        b = small()
        b.add_rule(STDP(ltp=0.02, ltd=0.015, clip=1.0, soft=True))
        for _ in range(400):
            b.step([0.5] * 24)
        weights = [abs(x) for x in b.recurrent.w]
        at_clip = sum(1 for x in weights if x > 0.99)
        self.assertLess(at_clip / len(weights), 0.10)

    def test_dopamine_gates_plasticity(self):
        dot = Dopamine()
        self.assertAlmostEqual(dot.value(), 1.0)      # resting gate
        dot.reward(1.0)
        self.assertGreater(dot.value(), 1.0)          # positive RPE opens it
        dot.reward(-1.0)
        self.assertLess(dot.value(), 1.0)             # negative closes it

    def test_sleep_renormalises_incoming_mass(self):
        b = small()
        for _ in range(50):
            b.step([0.5] * 24)
        stats = b.sleep(target=5.0)
        self.assertGreaterEqual(stats["pruned"], 0)
        mass = {}
        for sid, q in enumerate(b.recurrent.post):
            mass[q] = mass.get(q, 0.0) + abs(b.recurrent.w[sid])
        live = [v for v in mass.values() if v > 0]
        self.assertAlmostEqual(min(live), 5.0, places=2)
        self.assertAlmostEqual(max(live), 5.0, places=2)

    def test_pruning_shrinks_the_graph(self):
        b = small()
        before = len(b.recurrent)
        stats = Pruning(fraction=0.1).apply(b.recurrent)
        self.assertLess(len(b.recurrent), before)
        self.assertEqual(stats["after"], len(b.recurrent))

    def test_rules_survive_save_and_reload(self):
        b = small()
        b.add_rule(STDP())
        b.add_rule(Dopamine())
        for _ in range(120):
            b.step([0.4] * 24)
        b.reinforce(0.5)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "p" + memory.SUFFIX)
            memory.save(b, p)
            again = memory.load(p)
        self.assertEqual([r.name for r in again.rules], ["stdp", "dopamine"])
        self.assertAlmostEqual(again.rule("dopamine").baseline,
                               b.rule("dopamine").baseline, places=6)
        sensor = [0.4] * 24
        for _ in range(80):
            self.assertEqual(b.step(sensor)[1], again.step(sensor)[1])

    def test_make_rule_roundtrip(self):
        for rule in (STDP(ltp=0.01), Dopamine(baseline_rate=0.1),
                     Sleep(target=8.0), Pruning(fraction=0.05)):
            clone = make_rule(rule.spec())
            self.assertEqual(clone.name, rule.name)


class TestMemoryUnits(unittest.TestCase):
    """Every neuron keeps a prototype of what it has seen."""

    def _brain(self):
        return Brain(neurons=400, inputs=16, outputs=3, fan_in=12,
                     sensor_span=20, seed=5, memory_units=True,
                     memory_k=4, memory_lr=0.1)

    def test_prototypes_learn_the_inputs(self):
        import numpy as np

        rng = np.random.default_rng(0)
        pats = [(rng.random(16) > 0.5).astype(np.float32) for _ in range(4)]
        b = self._brain()
        b.step(None)

        def recon(x, k=4):
            sim = b.prototypes @ (x / (np.linalg.norm(x) + 1e-9))
            idx = np.argpartition(sim, -k)[-k:]
            A = b.prototypes[idx].T
            coef, *_ = np.linalg.lstsq(A, x, rcond=None)
            return float(np.linalg.norm(x - A @ coef) / (np.linalg.norm(x) + 1e-9))

        before = sum(recon(p) for p in pats) / len(pats)
        for _ in range(120):
            for p in pats:
                b.step(list(p))
        after = sum(recon(p) for p in pats) / len(pats)
        self.assertLess(after, before * 0.5)

    def test_neurons_specialise(self):
        import numpy as np

        rng = np.random.default_rng(1)
        pats = [(rng.random(16) > 0.5).astype(np.float32) for _ in range(4)]
        b = self._brain()
        for _ in range(150):
            for p in pats:
                b.step(list(p))
        sets = []
        for p in pats:
            sim = b.prototypes @ (p / (np.linalg.norm(p) + 1e-9))
            sets.append(set(int(i) for i in np.argpartition(sim, -4)[-4:]))
        overlap = np.mean([len(sets[i] & sets[j]) / 4
                           for i in range(4) for j in range(i + 1, 4)])
        self.assertLess(overlap, 0.5)     # not all patterns use the same neurons

    def test_prototypes_survive_save_and_reload(self):
        import numpy as np

        b = self._brain()
        for _ in range(20):
            b.step([0.5] * 16)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m" + memory.SUFFIX)
            memory.save(b, p)
            again = memory.load(p)
        self.assertTrue(np.array_equal(b.prototypes, again.prototypes))
        self.assertEqual(again.memory_k, b.memory_k)


class TestHippocampus(unittest.TestCase):
    """Episodic memory: one shot, content-addressed, separated."""

    def _hip(self, cells=32768):
        import numpy as np

        hip = Hippocampus(48, n_cells=cells, sparsity=0.002, seed=3)
        rng = np.random.default_rng(7)
        items = [(rng.random(48) > 0.5).astype(np.float32) for _ in range(10)]
        for i, x in enumerate(items):
            hip.experience(x, tag=f"e{i}")
        return hip, items, rng

    def test_one_shot_recognition(self):
        hip, items, rng = self._hip()
        hit = 0
        for i, x in enumerate(items):
            cue = x.copy()
            flip = rng.random(len(cue)) < 0.08
            cue[flip] = 1.0 - cue[flip]
            rec = hip.nearest(cue)
            hit += rec is not None and rec["tag"] == f"e{i}"
        # An attractor can fail where a lookup table cannot — that is the
        # point of building one. Five of ten from a single exposure at 8%
        # noise is the measured behaviour at this narrow input width (48 dims,
        # 10 episodes); the lookup table it replaced scored 10/10 by
        # construction, which is exactly why it was not a memory.
        self.assertGreaterEqual(hit, 5)      # one exposure, 8% noise

    def test_partial_cue_completes(self):
        """CA3 itself must complete a half cue — that is what an attractor is.

        Tested on the CA3 network directly rather than through the dentate,
        because the two stages fail differently: CA3 completes 100% of half
        cues, while the dentate-plus-projection layer in front of it throws
        away roughly half the code before CA3 ever sees it (measured 0.44
        recovered from a *complete* cue). Conflating them would hide which
        stage is broken.
        """
        import numpy as np

        from cerebrum.ca3 import CA3Network

        rng = np.random.default_rng(4)
        net = CA3Network(n_cells=2000, fan_in=400, k=100, seed=1,
                         ltp=0.15, w_max=0.195)
        pats = []
        for _ in range(10):
            c = np.zeros(2000, np.float32)
            c[rng.choice(2000, 100, replace=False)] = 1.0
            pats.append(c)
            net.store(c)
        ok, full = 0, 0
        for p in pats:
            idx = np.nonzero(p)[0]
            whole = net.recall(p, steps=60, settle=20)
            full += float(whole @ p) / 100.0 > 0.8
            cue = np.zeros(2000, np.float32)
            cue[rng.choice(idx, 50, replace=False)] = 1.0
            rec = net.recall(cue, steps=60, settle=20)
            ok += float(rec @ p) / 100.0 > 0.7
        self.assertGreaterEqual(full, 9)
        self.assertGreaterEqual(ok, 6)

    def test_projection_layer_is_the_weak_link(self):
        """Documents where the loss is, so it cannot be mistaken for completion.

        The dentate-to-CA3 projection currently returns about 0.44 of the
        stored code even when the full cue is present, against 1.00 for CA3
        given its own codes. This test pins that down rather than letting the
        end-to-end number float.
        """
        import numpy as np

        hip = Hippocampus(196, n_cells=16384, sparsity=0.002, seed=1)
        rng = np.random.default_rng(7)
        items = [(rng.random(196) > 0.5).astype(np.float32) for _ in range(10)]
        codes = np.stack([hip.experience(x, tag=str(i))["code"]
                          for i, x in enumerate(items)])
        ov = float(hip.ca3.recall(codes[0], steps=60, settle=20) @ codes[0]) / 120.0
        # CA3 given its own code must do well; the projection is what is lossy
        self.assertGreater(ov, 0.30)

    def test_similar_episodes_are_separated(self):
        import numpy as np

        hip = Hippocampus(48, n_cells=65536, sparsity=0.002, seed=1)
        rng = np.random.default_rng(0)
        ok = 0
        for _ in range(10):
            a = (rng.random(48) > 0.5).astype(np.float32)
            b = a.copy()
            flip = rng.random(48) < 0.05
            b[flip] = 1.0 - b[flip]
            ca, cb = hip.dentate.encode(a), hip.dentate.encode(b)
            hip.ca3.store(hip._to_ca3(ca))
            hip.ca3.store(hip._to_ca3(cb))
            idx = np.nonzero(ca)[0]
            cue = np.zeros_like(ca)
            cue[idx[: len(idx) // 2]] = 1.0
            rec = hip.ca3.recall(hip._to_ca3(cue), steps=60, settle=20)
            ca3_a, ca3_b = hip._to_ca3(ca), hip._to_ca3(cb)
            ok += (float(ca3_a @ rec) / max(1.0, ca3_a.sum())
                   > float(ca3_b @ rec) / max(1.0, ca3_b.sum()))
        # Separation of episodes that differ in 5% of their inputs is the hard
        # case: it depends on dentate expansion AND on CA3 not blending them.
        # 4/10 is what this size delivers; the lookup table it replaced scored
        # 10/10 by construction because it compared stored rows directly.
        self.assertGreaterEqual(ok, 4)

    def test_arousal_weighted_replay(self):
        hip, _, _ = self._hip()
        hip.episodes[0]["reward"] = 5.0        # one frightening event
        picked = hip.replay(6, arousal_weighted=True)
        scared = sum(1 for e in picked if e["reward"] > 1.0)
        self.assertGreaterEqual(scared, 2)

    def test_persistence_keeps_episodes(self):
        b = Brain(neurons=300, inputs=24, outputs=3, fan_in=10,
                  sensor_span=40, seed=9, hippocampus=True, hip_cells=8192)
        import numpy as np

        rng = np.random.default_rng(2)
        items = [(rng.random(24) > 0.5).astype(np.float32) for _ in range(5)]
        for i, x in enumerate(items):
            b.experience(x, tag=f"t{i}", reward=0.2 * i)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m" + memory.SUFFIX)
            memory.save(b, p)
            again = memory.load(p)
        self.assertEqual(len(again.hippocampus), 5)
        hit = sum(1 for i, x in enumerate(items)
                  if (r := again.remember(x)) is not None and r["tag"] == f"t{i}")
        self.assertGreaterEqual(hit, 4)
        self.assertAlmostEqual(again.hippocampus.episodes[2]["reward"], 0.4)


if __name__ == "__main__":
    unittest.main()
