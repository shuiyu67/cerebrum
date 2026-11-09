"""`.brain` memory files: a portable, versioned snapshot of a whole brain.

Layout — one ZIP container, no pickle, every part checksummed::

    manifest.json     format version, build parameters, signature, CRCs
    connectome.bin    recurrent synapses (pre, post, weight) as raw LE arrays
    sensor.bin        sensor projection (pre, post, weight)
    neurons.bin       membrane state, thresholds, biases, adaptation, resources
    scalars.json      homeostatic gain/drive and step counters

Design choices, and why they matter later:

* **No pickle.** A `.brain` file can be inspected, diffed and loaded without
  executing code. ``np.load(..., allow_pickle=True)`` on an untrusted file is
  remote code execution; a raw-array container is not.
* **Self-describing.** ``build`` in the manifest is exactly what
  :class:`~cerebrum.core.Brain` needs to reconstruct an identical scaffold, so
  loading never guesses. A signature mismatch is detected **before** any
  allocation, and reported rather than silently producing a different brain.
* **Split parts.** Weights and neuron state are separate streams. That is what
  makes merging two memories (P2) a per-part operation instead of a rewrite,
  and lets a partially-corrupt file be diagnosed instead of rejected whole.
* **Explicit version.** ``FORMAT`` is bumped whenever the layout changes;
  ``load`` refuses a newer file rather than misreading it.

Only the standard library is used: ``zipfile``, ``struct``, ``json``,
``hashlib``, ``zlib``.
"""

from __future__ import annotations

import array
import json
import struct
import sys
import time
import zipfile
import zlib
from typing import Any, Optional

FORMAT = 4
MAGIC = "cerebrum-brain"
SUFFIX = ".brain"

# Population fields written to neurons.bin, in this exact order.
NEURON_FIELDS = ("v", "threshold", "bias", "adapt", "refractory", "resources")


class MemoryError_(Exception):
    """Raised when a `.brain` file cannot be read as one."""


# ----------------------------------------------------------------------------
# raw array packing
# ----------------------------------------------------------------------------

def _has_numpy(seq) -> bool:
    return hasattr(seq, "tobytes") and hasattr(seq, "dtype")


def _pack_i32(seq) -> bytes:
    if _has_numpy(seq):
        return seq.astype("int32").tobytes()
    a = array.array("i", (int(x) for x in seq))
    if sys.byteorder == "big":
        a.byteswap()
    return a.tobytes()


def _pack_f32(seq) -> bytes:
    if _has_numpy(seq):
        return seq.astype("float32").tobytes()
    a = array.array("f", (float(x) for x in seq))
    if sys.byteorder == "big":
        a.byteswap()
    return a.tobytes()


def _unpack_i32(blob: bytes, count: int):
    a = array.array("i")
    a.frombytes(blob[: count * 4])
    if sys.byteorder == "big":
        a.byteswap()
    return list(a)


def _unpack_f32(blob: bytes, count: int):
    a = array.array("f")
    a.frombytes(blob[: count * 4])
    if sys.byteorder == "big":
        a.byteswap()
    return list(a)


def _encode_connectome(syn) -> bytes:
    """Layout: n_pre, n_post, nnz (uint32/uint32/uint64) then the arrays."""
    pre = list(syn.pre)
    post = list(syn.post)
    weight = list(syn.w)
    nnz = len(weight)
    head = struct.pack("<IIQ", syn.n_pre, syn.n_post, nnz)
    return head + _pack_i32(pre) + _pack_i32(post) + _pack_f32(weight)


def _decode_connectome(blob: bytes):
    n_pre, n_post, nnz = struct.unpack("<IIQ", blob[:16])
    body = blob[16:]
    pre = _unpack_i32(body, nnz)
    post = _unpack_i32(body[4 * nnz:], nnz)
    weight = _unpack_f32(body[8 * nnz:], nnz)
    return n_pre, n_post, pre, post, weight


def _encode_neurons(pop) -> bytes:
    """Layout: n (uint32), n_fields (uint16), then each field as float32[n]."""
    head = struct.pack("<IH", pop.n, len(NEURON_FIELDS))
    chunks = [head]
    for name in NEURON_FIELDS:
        chunks.append(_pack_f32(getattr(pop, name)))
    return b"".join(chunks)


def _decode_neurons(blob: bytes):
    n, n_fields = struct.unpack("<IH", blob[:6])
    body = blob[6:]
    out = {}
    for i in range(n_fields):
        out[NEURON_FIELDS[i]] = _unpack_f32(body[i * 4 * n:], n)
    return n, out


# ----------------------------------------------------------------------------
# save
# ----------------------------------------------------------------------------

def _build_spec(brain) -> dict:
    """Everything `Brain.__init__` needs to rebuild the same scaffold."""
    pop = brain.population
    return {
        "neurons": brain.n_neurons,
        "inputs": brain.n_inputs,
        "outputs": brain.n_outputs,
        "fan_in": brain.fan_in,
        "sensor_span": brain.sensor_span,
        "sensor_gain": brain.sensor_gain,
        "hippocampus": bool(getattr(brain, "hippocampus", None) is not None),
        "hip_cells": int(getattr(getattr(brain, "hippocampus", None), "dentate", None).n_cells)
        if getattr(brain, "hippocampus", None) is not None else 65536,
        "n_ca3": int(getattr(getattr(brain, "hippocampus", None), "n_ca3", 4000))
        if getattr(brain, "hippocampus", None) is not None else 4000,
        "memory_units": bool(getattr(brain, "memory_units", False)),
        "memory_k": int(getattr(brain, "memory_k", 32)),
        "memory_lr": float(getattr(brain, "memory_lr", 0.05)),
        "areas": getattr(brain, "areas", None),
        "seed": brain.seed,
        "backend": brain.backend_name,
        "neuron_params": dict(getattr(brain, "neuron_params", {})),
        "population": {
            "tau_m": pop.tau_m,
            "g_adapt": pop.g_adapt,
            "tau_adapt": pop.tau_adapt,
            "refractory_steps": pop.refractory_steps,
            "u_short_term": pop.u_short_term,
            "tau_recovery": pop.tau_recovery,
            "target_hz": pop.target_hz,
            "tau_homeostasis": pop.tau_homeostasis,
            "eta_homeostasis": pop.eta_homeostasis,
        },
    }


def save(brain, path: str) -> dict:
    """Write ``brain`` to ``path``. Returns the manifest that was written."""
    prev = list(brain._prev_spikes)
    parts = {
        "connectome.bin": _encode_connectome(brain.recurrent),
        "sensor.bin": _encode_connectome(brain.sensor_proj),
        "neurons.bin": _encode_neurons(brain.population),
        # the spike vector of the last step: without it, a reloaded brain
        # takes one step with no recurrent input and the trajectory shifts
        "spikes.bin": struct.pack("<I", len(prev)) + _pack_i32(prev),
    }
    pop = brain.population
    scalars = {
        "gain": float(pop.gain),
        "drive": float(pop.drive),
        "spike_total": int(brain._spike_total),
        "step_total": int(brain._step_total),
    }
    # plasticity rules and their traces are part of who this brain is
    rules = [r.spec() for r in getattr(brain, "rules", [])
             if hasattr(r, "spec")]
    # episodic memory: the events themselves, not just the state they left
    hip = getattr(brain, "hippocampus", None)
    if hip is not None:
        parts["dentate.bin"] = _encode_connectome_like(
            hip.dentate.proj, hip.dentate.bias)
        parts["episodes.bin"] = _encode_episodes(hip)
        scalars["hip_clock"] = hip._clock
        scalars["hip_cells"] = hip.dentate.n_cells
        scalars["hip_k"] = hip.dentate.k

    # per-neuron memories: this is literally what the brain remembers
    if getattr(brain, "memory_units", False):
        proto = brain.prototypes
        parts["prototypes.bin"] = (
            struct.pack("<II", proto.shape[0], proto.shape[1])
            + _pack_f32(proto.reshape(-1))
        )

    stdp = getattr(brain, "rule", lambda n: None)("stdp")
    if stdp is not None and stdp._pre_trace is not None:
        parts["traces.bin"] = (
            struct.pack("<I", int(brain.n_neurons))
            + _pack_f32(stdp._pre_trace) + _pack_f32(stdp._post_trace)
        )
    for r in getattr(brain, "rules", []):
        if getattr(r, "name", "") == "dopamine":
            scalars["dopamine_baseline"] = float(r.baseline)
            scalars["dopamine_signal"] = float(r._signal)

    manifest = {
        "magic": MAGIC,
        "format": FORMAT,
        "rules": rules,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "engine": _engine_version(),
        "spec": _build_spec(brain),
        "parts": {
            name: {"bytes": len(blob), "crc32": zlib.crc32(blob) & 0xFFFFFFFF}
            for name, blob in parts.items()
        },
        "scalars": scalars,
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        for name, blob in parts.items():
            zf.writestr(name, blob)
    return manifest


def _engine_version() -> str:
    try:
        from . import __version__

        return __version__
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------------
# inspect / load
# ----------------------------------------------------------------------------

def info(path: str) -> dict:
    """Read the manifest without building a brain."""
    with zipfile.ZipFile(path) as zf:
        raw = zf.read("manifest.json")
    return json.loads(raw)


def _read_parts(path: str) -> tuple[dict, dict]:
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        if "manifest.json" not in names:
            raise MemoryError_(f"{path}: not a .brain file (no manifest.json)")
        manifest = json.loads(zf.read("manifest.json"))
        if manifest.get("magic") != MAGIC:
            raise MemoryError_(f"{path}: wrong magic {manifest.get('magic')!r}")
        fmt = int(manifest.get("format", 0))
        if fmt > FORMAT:
            raise MemoryError_(
                f"{path}: format {fmt} is newer than this engine reads ({FORMAT})"
            )
        blobs = {}
        for name, meta in manifest.get("parts", {}).items():
            if name not in names:
                raise MemoryError_(f"{path}: missing part {name}")
            blob = zf.read(name)
            crc = zlib.crc32(blob) & 0xFFFFFFFF
            if crc != meta.get("crc32"):
                raise MemoryError_(
                    f"{path}: {name} is corrupt (crc {crc:08x} != {meta['crc32']:08x})"
                )
            blobs[name] = blob
    return manifest, blobs


def load(path: str, backend: Optional[str] = None):
    """Rebuild the brain stored in ``path``.

    Parameters
    ----------
    backend:
        Override the backend recorded in the file. Useful for pulling a
        NumPy-built memory into a NumPy-less environment (the numbers survive;
        the pure backend is simply slower).

    Raises
    ------
    MemoryError_
        If the file is not a `.brain`, is a newer format, or any part fails
        its checksum.
    """
    from .core import Brain

    manifest, blobs = _read_parts(path)
    spec = dict(manifest["spec"])
    pop_params = spec.pop("population", {}) or {}
    spec.pop("backend", None)
    spec.pop("neuron_params", None)

    # hippocampus sizing is applied when the Hippocampus object is built
    spec.pop("n_ca3", None)
    brain = Brain(backend=backend, **spec)
    brain._memory_manifest = manifest

    from .plasticity import make_rule

    for rule_spec in manifest.get("rules", []):
        try:
            brain.add_rule(make_rule(rule_spec))
        except ValueError:
            continue    # unknown rule from a newer engine: ignore, don't fail

    n_pre, n_post, pre, post, weight = _decode_connectome(blobs["connectome.bin"])
    if (n_pre, n_post) != (brain.n_neurons, brain.n_neurons):
        raise MemoryError_(
            f"connectome is {n_pre}x{n_post}, brain is "
            f"{brain.n_neurons}x{brain.n_neurons}"
        )
    _overwrite_connectome(brain.recurrent, pre, post, weight)

    n_pre, n_post, pre, post, weight = _decode_connectome(blobs["sensor.bin"])
    _overwrite_connectome(brain.sensor_proj, pre, post, weight)

    n, fields = _decode_neurons(blobs["neurons.bin"])
    if n != brain.n_neurons:
        raise MemoryError_(
            f"neuron state has {n} entries, brain has {brain.n_neurons}"
        )
    _overwrite_population(brain.population, fields)

    blob = blobs.get("spikes.bin")
    if blob:
        (count,) = struct.unpack("<I", blob[:4])
        prev = _unpack_i32(blob[4:], count)
        brain._prev_spikes = _assign(prev, brain._prev_spikes)

    blob = blobs.get("dentate.bin")
    if blob and getattr(brain, "hippocampus", None) is not None:
        import numpy as np
        from .hippocampus import Dentate, Hippocampus
        from .ca3 import CA3Network

        hip = brain.hippocampus
        proj, bias = _decode_connectome_like(blob)
        hip.dentate.proj = np.asarray(proj, np.float32).reshape(
            hip.dentate.n_cells, hip.dentate.n_inputs)
        hip.dentate.bias = np.asarray(bias, np.float32)

        ep = blobs.get("episodes.bin")
        if ep:
            tags, rewards, times, codes = _decode_episodes(ep)
            # Rebuild the attractor from its episodes instead of storing
            # 3.2 million recurrent weights. Every episode is re-written into
            # the recurrent matrix on load, which is the same one-shot Hebbian
            # write that happened when it was first experienced — so the
            # restored network completes identically and the file stays small.
            hip.ca3 = CA3Network(
                n_cells=getattr(hip, "n_ca3", hip.ca3.n_cells),
                fan_in=hip.ca3.fan_in,
                k=getattr(hip, "ca3_k", hip.ca3.k),
                seed=0, ltp=hip.ca3.ltp, w_max=hip.ca3.w_max,
            )
            hip.episodes = []
            for i, code in enumerate(codes):
                code_vec = np.zeros(hip.ca3.n_cells, np.float32)
                code_vec[np.asarray(code, np.int64)[
                    np.asarray(code, np.int64) < hip.ca3.n_cells
                ]] = 1.0
                hip.ca3.store(code_vec)
                hip.episodes.append({
                    "t": i, "wall": times[i], "tag": tags[i],
                    "reward": rewards[i],
                    "code": code_vec,
                    # the original input is not stored in the file; replayed
                    # episodes are codes, which is what recall works from
                    "x": np.zeros(brain.n_inputs, np.float32),
                })
        hip._clock = int(manifest.get("scalars", {}).get("hip_clock", len(codes)))

    blob = blobs.get("prototypes.bin")
    if blob and getattr(brain, "memory_units", False):
        import numpy as np

        rows, cols = struct.unpack("<II", blob[:8])
        values = _unpack_f32(blob[8:], rows * cols)
        brain.prototypes = np.asarray(values, np.float32).reshape(rows, cols)

    blob = blobs.get("traces.bin")
    stdp = brain.rule("stdp")
    if blob and stdp is not None:
        (count,) = struct.unpack("<I", blob[:4])
        span = count * 4
        stdp.load_state({
            "pre_trace": _assign_traces(_unpack_f32(blob[4:], count), stdp._pre_trace),
            "post_trace": _assign_traces(
                _unpack_f32(blob[4 + span:], count), stdp._post_trace),
        })

    gate = brain.rule("dopamine")
    if gate is not None:
        sc0 = manifest.get("scalars", {})
        gate.baseline = float(sc0.get("dopamine_baseline", gate.baseline))
        gate._signal = float(sc0.get("dopamine_signal", 0.0))

    sc = manifest.get("scalars", {})
    brain.population.gain = float(sc.get("gain", brain.population.gain))
    brain.population.drive = float(sc.get("drive", brain.population.drive))
    brain._spike_total = int(sc.get("spike_total", 0))
    brain._step_total = int(sc.get("step_total", 0))
    return brain


def _assign_traces(values, target):
    """Trace arrays keep whatever dtype the target uses."""
    if hasattr(target, "dtype"):
        import numpy as np

        return np.asarray(values, dtype=target.dtype)
    return list(values)


def _encode_connectome_like(matrix, bias) -> bytes:
    """A dense matrix plus a bias vector as raw float32."""
    rows, cols = matrix.shape
    head = struct.pack("<II", rows, cols)
    return (head + _pack_f32(matrix.reshape(-1)) + _pack_f32(bias))


def _decode_connectome_like(blob: bytes):
    rows, cols = struct.unpack("<II", blob[:8])
    body = blob[8:]
    matrix = _unpack_f32(body, rows * cols)
    bias = _unpack_f32(body[rows * cols * 4:], rows)
    return matrix, bias


def _encode_episodes(hip) -> bytes:
    """Episode metadata as JSON, codes as raw int32."""
    import numpy as np

    rows = [np.nonzero(e["code"])[0] for e in hip.episodes]
    tags = [e.get("tag") for e in hip.episodes]
    rewards = [float(e.get("reward", 0.0)) for e in hip.episodes]
    times = [float(e.get("wall", 0.0)) for e in hip.episodes]
    meta = json.dumps({"tags": tags, "rewards": rewards, "times": times,
                       "k": int(rows[0].size) if rows else 0}).encode()
    head = struct.pack("<I", len(meta))
    body = bytearray(head + meta)
    for r in rows:
        body += _pack_i32(r)
    return bytes(body)


def _decode_episodes(blob: bytes):
    import numpy as np

    (mlen,) = struct.unpack("<I", blob[:4])
    meta = json.loads(blob[4:4 + mlen])
    body = blob[4 + mlen:]
    k = meta["k"]
    codes = []
    for i in range(len(meta["tags"])):
        codes.append(_unpack_i32(body[i * 4 * k:], k))
    return meta["tags"], meta["rewards"], meta["times"], codes


def _assign(values, target):
    """Return ``values`` in the array type ``target`` already uses."""
    if hasattr(target, "dtype"):
        import numpy as np

        return np.asarray(values, dtype=np.int32)
    return list(values)


def _overwrite_connectome(syn, pre, post, weight) -> None:
    """Replace every synapse in ``syn`` while keeping its identity."""
    try:
        syn._pre[: len(pre)] = pre
        syn._post[: len(post)] = post
        syn._w[: len(weight)] = weight
        syn._n = len(weight)
        syn._dirty = True
    except AttributeError:
        # pure backend keeps three plain lists
        syn.pre[:] = pre
        syn.post[:] = post
        syn.w[:] = weight
        syn._out = [[] for _ in range(syn.n_pre)]
        for sid, p in enumerate(pre):
            syn._out[p].append(sid)


def _overwrite_population(pop, fields) -> None:
    for name, values in fields.items():
        cur = getattr(pop, name)
        try:
            cur[:] = values
        except (TypeError, ValueError):
            setattr(pop, name, values)


def peek(path: str) -> str:
    """One-line human summary of a memory file."""
    m = info(path)
    spec = m["spec"]
    return (
        f"{path}: neurons={spec['neurons']:,} sensors={spec['inputs']} "
        f"actuators={spec['outputs']} fan_in={spec['fan_in']} "
        f"seed={spec['seed']} backend={spec.get('backend', '?')} "
        f"format={m['format']} created={m['created']}"
    )


# ----------------------------------------------------------------------------
# merge / graft  (P2)
# ----------------------------------------------------------------------------

def _experience(brain) -> float:
    """How much a brain has lived, used as its merge weight."""
    return float(max(1, brain._step_total))


def _structure_key(brain):
    return (tuple(brain.recurrent.pre), tuple(brain.recurrent.post))


def merge(brains, weights=None, backend: Optional[str] = None):
    """Fuse several memories into a new brain.

    The merge is a weighted element-wise average of every synapse and every
    neuron parameter, with the brain's step count as the default weight — a
    brain that has lived twice as long contributes twice as much. Thresholds
    are averaged, not summed, so the result fires at the same scale as its
    inputs instead of saturating.

    All inputs must share the same **connectome structure** (same pre/post
    arrays, i.e. built from the same seed and shape). That is checked, not
    assumed: averaging weights across different wirings would be meaningless.

    Parameters
    ----------
    brains:
        Two or more :class:`~cerebrum.core.Brain` instances.
    weights:
        Optional per-brain weights. Defaults to each brain's step count.
    """
    if len(brains) < 2:
        raise ValueError("merge needs at least two brains")

    base = brains[0]
    key = _structure_key(base)
    for k, other in enumerate(brains[1:], start=1):
        if (other.n_neurons, other.n_inputs, other.n_outputs) != (
            base.n_neurons, base.n_inputs, base.n_outputs
        ):
            raise MemoryError_(
                f"brain {k} has shape {(other.n_neurons, other.n_inputs, other.n_outputs)}, "
                f"expected {(base.n_neurons, base.n_inputs, base.n_outputs)}; "
                "use graft() for different sizes"
            )
        if _structure_key(other) != key:
            raise MemoryError_(
                f"brain {k} has a different connectome structure "
                "(different seed or fan-in); merge needs identical wiring"
            )

    if weights is None:
        weights = [_experience(b) for b in brains]
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("merge weights must sum to a positive number")
    frac = [w / total for w in weights]

    out = _clone(base, backend=backend)
    _blend_into(out, brains, frac)
    return out


def _clone(brain, backend: Optional[str] = None):
    """A structurally identical, state-copied brain."""
    from .core import Brain

    copy = Brain(
        neurons=brain.n_neurons,
        inputs=brain.n_inputs,
        outputs=brain.n_outputs,
        fan_in=brain.fan_in,
        sensor_span=brain.sensor_span,
        sensor_gain=brain.sensor_gain,
        backend=backend or brain.backend_name,
        seed=brain.seed,
        **getattr(brain, "neuron_params", {}),
    )
    _overwrite_connectome(copy.recurrent, list(brain.recurrent.pre),
                          list(brain.recurrent.post), list(brain.recurrent.w))
    _overwrite_connectome(copy.sensor_proj, list(brain.sensor_proj.pre),
                          list(brain.sensor_proj.post), list(brain.sensor_proj.w))
    _overwrite_population(copy.population,
                          {f: list(getattr(brain.population, f)) for f in NEURON_FIELDS})
    copy.population.gain = brain.population.gain
    copy.population.drive = brain.population.drive
    copy._spike_total = brain._spike_total
    copy._step_total = brain._step_total
    return copy


def _blend_into(out, brains, frac) -> None:
    """Write the weighted average of ``brains`` into ``out`` (same structure)."""
    for target, attr in ((out.recurrent, "recurrent"), (out.sensor_proj, "sensor_proj")):
        acc = [0.0] * len(target.w)
        for f, b in zip(frac, brains):
            src = getattr(b, attr).w
            for i, v in enumerate(src):
                acc[i] += f * v
        target.w[:] = acc

    pop = out.population
    for name in NEURON_FIELDS:
        acc = [0.0] * pop.n
        for f, b in zip(frac, brains):
            src = getattr(b.population, name)
            for i, v in enumerate(src):
                acc[i] += f * v
        getattr(pop, name)[:] = acc

    pop.gain = sum(f * b.population.gain for f, b in zip(frac, brains))
    pop.drive = sum(f * b.population.drive for f, b in zip(frac, brains))
    out._step_total = sum(b._step_total for b in brains)
    out._spike_total = sum(b._spike_total for b in brains)


def graft(source, target, weight: float = 1.0):
    """Transplant ``source``'s memory into ``target``'s scaffold.

    Sizes may differ. Neuron and sensor indices are mapped proportionally
    (``dst = src * n_dst // n_src``), so a 500-neuron memory lands in the same
    *relative* place inside an 800-neuron brain. Synapses whose endpoints both
    survive are kept and accumulated; synapses mapping outside the target are
    dropped. Returns a **new** brain; both inputs are left untouched.

    This is lossy by construction — the point is that experience is carried
    over, not that the donor is reproduced bit for bit.
    """
    out = _clone(target)

    def scale_index(i: int, n_src: int, n_dst: int) -> int:
        return min(n_dst - 1, int(i) * n_dst // max(1, n_src))

    # recurrent: accumulate remapped synapses into the existing wiring
    dst_w = out.recurrent.w
    dst_w[:] = [v * (1.0 - weight) for v in dst_w]
    pre = source.recurrent.pre
    post = source.recurrent.post
    src_w = source.recurrent.w
    n_src = source.n_neurons
    n_dst = out.n_neurons
    index = {}
    for sid in range(len(out.recurrent.w)):
        index[(out.recurrent.pre[sid], out.recurrent.post[sid])] = sid
    for sid in range(len(src_w)):
        p = scale_index(pre[sid], n_src, n_dst)
        q = scale_index(post[sid], n_src, n_dst)
        at = index.get((p, q))
        if at is not None:
            dst_w[at] += weight * src_w[sid]

    # neuron state: blend over the mapped overlap
    pop = out.population
    for name in NEURON_FIELDS:
        dst = getattr(pop, name)
        src = getattr(source.population, name)
        blended = [v * (1.0 - weight) for v in dst]
        for i, v in enumerate(src):
            blended[scale_index(i, n_src, n_dst)] += weight * v
        dst[:] = blended

    pop.gain = (1.0 - weight) * pop.gain + weight * source.population.gain
    pop.drive = (1.0 - weight) * pop.drive + weight * source.population.drive
    return out
