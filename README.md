# cerebrum v2

A spiking brain engine, and — in `cerebrum.brain` — an assembled one: neuron
sheets with positions, a thalamic gate, a three-layer cortex that forms its own
maps, basal ganglia that pick an action, a hippocampus that completes a
partial memory, an amygdala that learns fear in one trial, a cerebellum that
learns timing from error, a hypothalamus that supplies the drives, and five
neuromodulators that change how the rest computes.

LIF neurons, sparse event-driven wiring, inhibitory interneurons, and a memory
file you can save, reload, merge and transplant.

Not a classifier. Not a transformer. No loss function, no backward pass, no
gradient anywhere in the engine. Every mechanism in here changes one thing:
synaptic weight.

```python
from cerebrum.core import Brain
from cerebrum.plasticity import Eligibility, Dopamine
from cerebrum import memory

brain = Brain(neurons=139_000, inputs=363, outputs=3, fan_in=360, seed=1)
brain.add_rule(Eligibility())          # spike timing tags a synapse
brain.add_rule(Dopamine())             # reward cashes the tag in
brain.run(2500)                        # no input: it wakes itself up

action, spikes = brain.step(sensors)   # 363 sensors -> 3 actuators

memory.save(brain, "day1.brain")       # full state, checksummed
same = memory.load("day1.brain")       # bit-identical continuation
```

Runs on the standard library alone. NumPy is used automatically when present.

---

## The whole brain

Everything above is a substrate: one population, one wiring scheme, one output
readout. It can be driven, it can learn, and it cannot decide anything, because
a brain that is one homogeneous sheet has nowhere for a decision to happen.

`cerebrum.brain` is the assembled organ set. Each anatomical system is a
separate module with its own cells, its own dynamics and its own job, wired
into the loops that make behaviour possible.

```python
from cerebrum.brain import WholeBrain, Body

brain, body = WholeBrain(seed=0), Body(seed=0)
sensors = body.reset()
for _ in range(10_000):
    out = brain.step(sensors, us=1.0 if body.fear > 0.85 else 0.0)
    sensors, reward, done, info = body.act(out["action"])
    if reward:
        brain.reinforce(reward)          # dopamine = reward - prediction
    if done:
        sensors = body.reset()
```

| System | Module | Level | What it does |
|---|---|---|---|
| Neuron sheet | `sheet` | spiking | 2-D positions, distance-decayed E/I wiring, divisive inhibition, multiplicative homeostasis |
| Thalamus | `thalamus` | spiking | relay + reticular gate; the cortex suppresses what it is not tracking |
| Cortex | `cortex` | spiking | L4 → L2/3 → L5, self-organising thalamocortical map, prefrontal working memory |
| Basal ganglia | `basal_ganglia` | rate | D1/D2/GPe/STN/GPi, action selection by disinhibition |
| Hippocampus | `hippocampus` | hybrid | EC → DG → CA3 → CA1, one-shot storage, pattern completion, reinstatement |
| Amygdala | `amygdala` | spiking | LA/BLA/CE, one-trial fear conditioning gated by arousal |
| Hypothalamus | `hypothalamus` | rate | hunger, fatigue, fear, orexin — the drives that give behaviour a reason |
| Cerebellum | `cerebellum` | spiking | granule → Purkinje → deep nuclei, timing learned by climbing-fibre LTD |
| Neuromodulators | `neuromodulation` | scalar | dopamine, serotonin, noradrenaline, acetylcholine, orexin |

The level of description differs by system on purpose and every module says so
in its own docstring. Selection, drive and concentration are settling
processes and slow variables; modelling them as spikes costs time and adds no
mechanism. The sheets, thalamus, amygdala, hippocampus and cerebellum are
spiking.

### What each system is for, and how you can tell

Every claim below comes from an ablation in `cerebrum.brain.protocols`: the
same seed, one mechanism switched off, everything else identical.

| Claim | Control | Intact |
|---|---|---|
| Cortex partitions itself into areas without being told where | 0.499 adjacency, 144 fragments (winner-only learning) | **0.956 adjacency, 5 territories** (with a neighbourhood) |
| The thalamus gates what the cortex ignores | 100% of the distractor relay survives with the reticular nucleus lesioned | **57% survives**; the attended channel keeps 100% |
| The basal ganglia pick one action | 6 channels open, strongest proposal chosen 13% of the time (chance) | **1 channel open, chosen 100%** |
| The hippocampus completes a partial cue | 0.22 Jaccard with no storage | **0.69 Jaccard** after one exposure each |
| The amygdala learns fear in one trial | 0.00 with arousal blocked | **0.52 to the paired cue, 0.00 to an unpaired one** |
| The cerebellum learns timing from error | Purkinje response 0.99 with LTD disabled | **0.72 with LTD** — the trained response is suppressed |
| Dopamine codes prediction error, not reward | — | **1.00 → 0.68** as the reward becomes predicted, **−0.33 on omission** |

### The finding that made the architecture possible

The previous generation could not self-organise, and the reason was not the
learning rule. It was that its neurons had no positions.

Measured on an 800-neuron flat population with competitive input learning:
adjacent neurons agreed on their preferred stimulus 51% of the time against a
50% chance baseline. Preferences were scattered noise, and no learning rule
could have fixed it, because "adjacent" had no meaning.

Give the same cells coordinates, distance-decayed wiring, local inhibition and
a neighbourhood kernel, and the same competitive rule produces territories:

```
sigma = 0.00 (winner only)   adjacency 0.499   144 fragments
sigma = 0.25 (neighbourhood) adjacency 0.956     5 fragments
```

No region borders were specified anywhere. The areas are a consequence of the
geometry, which is also why they cannot appear without it.

### Where the assembled brain actually stands

It runs a closed loop at about 3 ms per millisecond of simulated time, all
systems engaged, no numerical blow-ups over tens of thousands of steps. On the
foraging task it eats, avoids the predator some of the time, and rests when
tired.

It is not competent. Over 8,000 milliseconds (three seeds) it eats 8-10 times,
is caught by the predator 200+ times, and spends most of its time resting and
fleeing. At these timescales the cortex's synapses have barely moved, so what
it does is reflex: thalamic relay, hypothalamic drive, basal ganglia gating.
Learning is present and measurable in every subsystem; it has not yet had time
to show up as skilled behaviour, and the cortical motor map — which L5 block
means which action — is fixed rather than learned, so the cortex proposes
weakly and the drives decide. That is the honest state of it.

---

## A million neurons as a language model

`cerebrum.lm` runs a million spiking neurons as a character language model.
The pool is random and fixed; only a linear readout is trained, online, in a
single pass.

```bash
python -m cerebrum.lm.train --neurons 1000000 --tokens 20000 --steps 2 --bins 512
python -m cerebrum.lm.tasks          # delayed recall
```

**Why a reservoir.** This engine has no backward pass and is not going to get
one — every mechanism in it changes a weight locally. So "train an LM" has to
mean something a gradient-free network can do. A reservoir is the honest
answer: the million neurons are never trained, and the only gradient anywhere
is one linear layer, the same amount a logistic regression uses.

**Scaling.** The assembled brain uses dense `n x n` matrices per region, which
is right at 900 cells and impossible at a million (10^12 floats, four
terabytes). The LM uses synapses grouped by presynaptic cell — each neuron
owns 32 contiguous outgoing synapses, so an index range is a slice and nothing
is ever sorted — with propagation as one scatter over the cells that fired.
A million neurons and 32M synapses fit in 253 MB and run at 18 characters per
second on one core.

### Result

Character-level, on this repository's own source (370k characters, 78-symbol
vocabulary at the time of measurement), 20,000 characters of training, one
pass, no epochs. The corpus is not committed — it is rebuilt from the tree on
first use, so the exact character count drifts as the repository grows:

| | bits/char | top-1 |
|---|---|---|
| uniform | 6.29 | — |
| bigram, closed form, same 20k | **4.36** | — |
| reservoir LM, online | 4.68 | 18.7% |

**It loses to a bigram by 0.33 bits.** That is the result, and the interesting
part is knowing why.

### Two real bugs, both found by measuring rather than reading

**1. The recurrent weights were about a hundred times too small.** The membrane
update multiplies synaptic current by `1/tau`, so at the physiologically
sensible `w = 0.1` one input moves the membrane by 0.005 and thirty of them
have to land in the same millisecond to matter. The pool was, in effect,
disconnected: measured echo after the input stopped was **exactly zero** at
every fan-out tried, up to 128 synapses per cell. At `w = 8` the same pool
reverberates for 5-10 ms and settles at a stable 10 Hz.

**2. The evaluation window was off by one.** Training fed `stream[t-k:t]` plus
the current character; the evaluation fed `stream[pos-k:pos]`, which includes
it. The model was handed an input it had never seen. Accuracy was 0% and the
loss *above* the uniform baseline — a signature that looks like "the
architecture cannot do this" and is actually an indexing error.

### What the pool is worth

Paired ablation, 20k pool, 6,000 characters, identical readout, pool features
switched off:

| | bits/char |
|---|---|
| readout without pool features | 4.879 |
| readout with pool features | **4.593** |

The pool contributes 0.29 bits. It is doing something. It is not doing enough
to beat a counting table.

### Delayed recall: what the architecture is actually for

A bigram cannot answer "which symbol did you see D steps ago" — it sees the
cue and nothing else, so 2.99 bits is its floor. A reservoir that integrates
should answer it.

| delay | bigram acc | model acc | chance |
|---|---|---|---|
| 3 | 14.4% | **17.6%** | 12.5% |
| 6 | 14.4% | 13.6% | 12.5% |
| 10 | 14.4% | 16.0% | 12.5% |

Barely above chance. The pool's memory is **2-3 characters**, measured, and
the task needs 3-10.

### Memory capacity: how many past inputs the pool actually holds

The delayed-recall result raised an obvious question — is the memory limited by
the number of synapses per cell, or by something else? Rather than guess, the
pool's capacity was measured directly with Jaeger's memory capacity:

```
MC_k = corr²( u(t-k), readout(state_t) )      MC = sum_k MC_k
```

A memoryless system scores about 1 (it reports the current input only). A delay
line of length D scores D. The control arm sets `w_scale = 0`, which removes
recurrence entirely.

```bash
python -m cerebrum.lm.capacity           # shape comparison at fixed synapse budget
python -c "from cerebrum.lm.capacity import sweep; sweep()"   # parameter sweep
```

| configuration (all 32M synapses) | total MC | depth | MC₁ |
|---|---|---|---|
| control, no recurrence | 0.06 | 0 | 0.01 |
| 1.0M neurons × 32 | **0.53** | 2 | 0.33 |
| 0.5M neurons × 64 | 0.44 | 1 | 0.32 |
| 0.25M neurons × 128 | 0.50 | 1 | 0.34 |
| 0.12M neurons × 256 | 0.45 | 1 | 0.29 |

**Eight times the synapses per cell buys nothing.** The weights were normalised
by `1/sqrt(fan_out)` to hold the spectral radius fixed, so the dynamics were
unchanged — which is the point: fan-out is not the lever.

So the lever was swept instead:

| knob | setting | total MC | depth |
|---|---|---|---|
| spectral radius | w = 6-8 (peak) | 0.64-0.67 | **1** |
| spectral radius | w = 16 (saturated) | 0.21 | 1 |
| adaptation | off | 0.61 (vs 0.33 at w=4) | 1 |
| membrane τ | 40 ms | 0.39 | 1 |
| membrane τ | 80 ms | 0.43 | 1 |
| balanced regime | w ≤ 1.2, strong inhibition | 0.05-0.12 | 0 |

**No configuration reached MC₂ above 0.20, or a depth beyond four.** Over
three seeds the shipped pool holds **two to three** inputs (12-18 ms); the best
rules found reach three to four. Nothing reached the ten-plus that language
needs.

That is not a tuning failure. A cell here needs two or three coincident inputs
to cross threshold, so it fires on the current volley instead of integrating
over several. Spike-frequency adaptation then suppresses the cells that were
just active, which is why switching it off doubles the single-step capacity
while leaving the depth at one.

### Why one, specifically

The pool is deterministic, so its dynamics can be probed exactly: take two
identical copies, give one an extra millimetre of input, then run both in
silence and watch the separation. It grows. The largest Lyapunov exponent is
positive from `w ≈ 1.3` upward and rises monotonically:

| w | λ (per ms) | rate | capacity | depth |
|---|---|---|---|---|
| 1.2 | −0.001 | 13 Hz | 0.32 | 1 |
| 2.2 | +0.029 | 17 Hz | 0.63 | 2 |
| 6.0 | +0.096 | 45 Hz | 0.67 | 1 |
| 12 | +0.150 | 108 Hz | 0.32 | 1 |

At the shipped weight the *direction* of the separation decorrelates within a
millisecond (cosine 0.46 after 1 ms, 0.13 after 6 ms). A chaotic network does
not hold a state and release it later; it amplifies any difference until it
saturates and replaces it with an unrelated one.

**A global inhibitory pool barely touches it — until it overshoots.** The
exponent stays positive as the inhibitory gain goes from 2 to 320 (at 160 it is
still +0.05), and it is unmoved by removing adaptation, by the membrane time
constant going 20 -> 80 ms, by fan-out going 32 -> 256, and by removing the
inhibitory dead zone (`target_frac = 0`). Push the gain to 800 and it does
cross zero — and capacity falls from 0.80 to 0.28 at the same time. The next
section measures that trade-off directly.

Two consequences worth stating plainly:

- **There is no window that is both stable and informative.** At the chaos
  boundary the pool is stable but runs at 13 Hz with capacity 0.32; by the time
  capacity peaks the network is already chaotic. The two requirements are met
  on opposite sides of the same line.
- **The best point found** is `w = 6, ginh = 2, target_frac = 0`: capacity
  0.92 at depth 3, still chaotic but with a longer decorrelation time. That is
  where to push, not synapse count.

What would actually give long memory is either designed rather than random
connectivity (the hippocampal module in `cerebrum/brain` does exactly this, and
completes patterns at 0.69 from a partial cue), or plasticity that moves the
spectral radius below the chaos threshold.

**This also closes the loop on the language model.** A pool with two or three
inputs of memory *is* a bigram with a short smear on it — which is exactly what
was measured:
4.68 bits against 4.36 for a counting table, and recall that dies at k=3. The
two results are the same fact seen from two directions.

Three measurement bugs had to be fixed before any of this measured anything,
and each one had produced a plausible-looking zero:

1. **Input coding.** Uniform current locks every driven cell into the same
   refractory phase, so the spike count tracks the phase and not the input
   (k=0 reconstruction 0.03). Rank coding — each input cell has its own fixed
   threshold — gives 0.85 for the same pool.
2. **Input coverage.** 2% of the pool driven left 54 cells firing per step at
   100k neurons: 0.05% activity, pure noise in the readout.
3. **Hold time.** One input per millisecond against a 20 ms membrane is an
   aliased signal. Holding each input for 6 ms is what makes the control arm
   score 0.85 instead of 0.04.

### The exponent does cross zero, and that is the bad news

Pushing inhibition far past the range examined earlier takes the pool across
the chaos boundary. It just does not take capacity with it:

| ginh | rate | λ | capacity | depth |
|---|---|---|---|---|
| 20 | 21.4 Hz | **+0.083** | **0.80** | 2 |
| 320 | 8.5 Hz | +0.013 | 0.36 | 1 |
| 800 | 7.3 Hz | **−0.012** | 0.28 | 1 |
| 3200 | 3.7 Hz | −0.010 | 0.13 | 0 |

A stable pool exists. It holds a third as much. Capacity falls monotonically
from the chaotic side to the stable side, so there is no window that is both
stable and informative, and the peak sits *slightly on the chaotic side*
(+0.07 to +0.08). That is the edge-of-chaos result measured in this pool
rather than imported from the literature.

Strong weights do not buy the memory back at λ = 0: `w = 6` scores 0.68 while
chaotic (+0.065) and 0.24 once inhibited to −0.006; `w = 10` gives 0.34 at
+0.035 and 0.28 at −0.006. Standing on the λ = 0 line costs two thirds of the
capacity every time.

Two structural candidates were tried, and both failed:

- **In-degree regularity.** The default Erdos-Renyi graph has Poisson
  in-degree (mean 64, sd 8.0, range 36-98), so the top of the tail is driven
  much harder than the mean cell and diverges first. Replacing it with a
  directed random-regular graph (sd 0.00 — every cell exactly 64 in and 64
  out) moved λ from +0.083 only to +0.077 and *cut* capacity from 0.80 to
  0.48. The tail is not the cause.
- **Dale's law.** Giving every cell one sign (80% excitatory, 20% inhibitory)
  instead of drawing each synapse independently drove the pool to 494 Hz:
  80/20 leaves net excitation at 0.6 of total drive. Correct biology, wrong
  budget — it needs inhibition the global pool does not supply.

### The pool contributes almost nothing to the language model

Measured by removing the recurrence entirely:

| w_scale | recurrence | bits/char |
|---|---|---|
| 0.0 | none | 5.363 |
| 2.0 | weak | 5.355 |
| 5.66 | the operating point | 5.311 |

The whole contribution of the recurrent pool is **0.05 bits**. With the
recurrence off the model is still 0.71 bits worse than a bigram table, which
puts the bottleneck outside the pool: one online pass of SGD over a
96 x 736 readout cannot match exact counting. Integrating the input longer
does not fix it — 4.864 (2 ms), 4.778 (6 ms), 4.720 (12 ms), 4.701 (16 ms),
saturating near 4.70 against the bigram's 4.431.

Heterogeneous input drive was the last candidate and it failed as well. Every
driven cell gets the *same* current, so an input group fires in lockstep; the
capacity measurement hit this and fixed it with per-cell thresholds.
Applying that fix here — a fixed lognormal per-cell gain, one float per
neuron — moved bits from 5.311 to 5.251 at gain spread 0.8. Nearly nothing,
because each (position, character) already drives its own disjoint cell set:
the pattern is carried by *which* cells fire and the phase adds no
information.

Read together: capacity one to two, a pool worth 0.05 bits, and a stable
region that remembers less than the chaotic one. The random reservoir is not
under-tuned, it is at its structural limit, and the limit is the single-spike
trigger — a cell needs two or three coincident inputs to fire, so it never
integrates and anything older than a few milliseconds has been overwritten.

### Learning made it worse, three times

The pool is frozen at initialisation, which is why the readout is the only
thing that ever learns and why the pool is worth 0.05 bits. Three local rules
were added to change that. All three were measured against a frozen control
that reproduced to the digit (+0.083 / capacity 0.80 / cv 0.109 on three
successive probes), so the numbers are clean.

| rule | lambda | capacity | in-norm cv |
|---|---|---|---|
| frozen control | +0.083 | **0.80** | 0.109 |
| synaptic scaling, outgoing | +0.221 | 0.61 | 0.873 |
| synaptic scaling, incoming | +0.223 | 0.61 | 0.873 |
| STDP + hard in-norm, lr 0.002 | +0.079 | 0.61 | 0.000 |
| STDP + hard in-norm, lr 0.01 | +0.062 | 0.48 | 0.000 |

Two failure modes, pointing at the same thing.

**Scaling on the wrong side runs away.** Acting on outgoing synapses controls
the neighbours, not the cell: 23 Hz -> 135 Hz within 1500 steps, output norm
23.9 -> 86.9 (the clip ceiling times fan-out), capacity 0.80 -> 0.13. A silent
cell had its outgoing weights amplified to the maximum and then fired a volley
into its targets. Moving the rule to incoming synapses fixed the direction and
not the outcome — the spread grew from cv 0.11 to 0.87, because cells that
never fired were amplified to the ceiling while cells that fired constantly
were cut to nearly nothing.

**Enforcing uniformity destroys the capacity.** The Song-Abbott constraint holds
every cell's incoming norm fixed, which is what keeps the exponent from moving
(+0.083 -> +0.062 rather than exploding). It also drives the norm spread to
exactly cv 0.000, and capacity falls to 0.48. Output normalisation did the same
earlier: 0.80 -> 0.48. A random pool draws its capacity from heterogeneity —
different cells happen to be sensitive to different input histories — and every
rule that cleans that heterogeneity up removes the very thing being read out.

The honest summary: **an unsupervised local rule cannot know which
heterogeneity to keep.** It can only enforce a statistic (a rate, a norm), and
every statistic it can enforce is orthogonal to the one that matters.

What that leaves is a rule that is local in its *mechanism* but not in its
*objective*: a three-factor rule, where a per-synapse eligibility trace from
STDP is multiplied by one global scalar — the readout's prediction error. The
trace stays local; the scalar is a single number for the whole pool. That is
the only variant here that could carry task information into the synapses
without a backward pass.

### The third factor is not enough either

The rule above has no idea what the pool is for, so the obvious next step is
to give it one: accumulate a per-synapse eligibility trace from STDP, then
commit that trace gated by a single scalar — how much better or worse the
prediction was than a running baseline. Local mechanism, global objective,
no gradient through the recurrence.

It does not work. Same pool, same corpus, same seed, only the scalar gain
changes:

| three_factor | bits/char |
|---|---|
| 0 (frozen pool) | **5.311** |
| 1 | 5.294 |
| 5 | 5.306 |
| 20 | 5.330 |
| 200 | 5.369 |

The best result is the frozen pool, and everything above the smallest gain is
worse than doing nothing. The improvement at gain 1 is 0.017 bits, which is
inside the seed-to-seed scatter of this measurement.

Why one scalar cannot do it, stated as simply as possible: the eligibility
trace answers *who was involved*, and the error answers *whether the outcome
was good*. It never answers *which of the synapses that were involved is to
blame*. Multiply a recency record by a single number and every synapse that
happened to fire recently gets pushed in the same direction — including the
ones that were irrelevant. At small gain nothing moves; at large gain the
random push destroys the heterogeneity the capacity was coming from, which is
why the curve is monotone in the wrong direction.

This is the classic credit-assignment problem, and it is the reason
three-factor rules in the literature are demonstrated on tasks with a handful
of synapses or a handful of choices, not on sequence models.

**What the brain does differently is not a bigger scalar.** Cortex carries
prediction error between layers — a signal with structure, not one number for
the whole population. A single random pool with one global modulator is a
degenerate case that has the shape of learning without its content.

So the honest position after four experiments: within this architecture,
nothing local beats freezing the pool, and every candidate that looked like it
should help either was directionally wrong (outgoing scaling), destroyed the
heterogeneity it was reading out (normalisation), or lacked the information to
assign credit (one scalar). The pool is a fixed random projection, and the
0.05 bits it contributes is what a fixed random projection is worth on this
task.

### What this means

The engine runs a million neurons and learns online, in one pass, with no
backpropagation, and the number is reproducible. It is not competitive with a
n-gram at this data budget, and the capacity measurement says why in a way that
does not depend on the corpus: **the pool holds two to three inputs, and no
parameter setting gets it past four.**

Two or three inputs of memory is a bigram with a short smear. That is enough to
beat a uniform baseline and to contribute 0.29 bits over its own readout, and
not enough to beat a counting table that gets the bigram exactly right.

An earlier version of this section guessed that the fix was more synapses per
cell. That was measured and it is wrong: eight times the fan-out at the same
total synapse count leaves the depth at one, and sweeping the spectral radius,
adaptation and membrane time constant does not move it either. What is missing
is not depth of wiring but the operating regime — a cell that needs two
coincident inputs fires on the current volley instead of integrating over
several. Memory of 10-20 steps out of this family of networks needs a balanced
regime with active gain control, which is a different piece of work from
scaling this one up.

## The substrate engine

The original engine — `cerebrum.core.Brain` — is still here and still what the
scale benchmarks measure. It is one population with one wiring scheme and one
readout: useful as a substrate, insufficient as a brain.

```mermaid
flowchart TB
    S["感觉层<br/>局部视野 · 感受野投射"] --> C
    subgraph C["皮层"]
        E["兴奋性神经元<br/>泄漏整合发放 · 突触电流有时程"]
        I["抑制性 20%<br/>负权重反馈"]
        P["可塑规则<br/>STDP · 资格迹 · 稳态 · 稀疏竞争"]
    end
    C --> D["齿状回<br/>稀疏扩张 · 相似经历分开"]
    D --> M["两条通路<br/>苔藓纤维稀疏 + 内嗅直投稠密"]
    M --> A["CA3 吸引子<br/>递归 20% · 一次写入 · 半线索补全"]
    A --> O["动作层<br/>三组投票 · 无分类器"]
    O -.->|动作改变世界| W["世界"]
    W -.->|奖励 → 多巴胺| P
    K["内核<br/>稀疏事件驱动 · numpy / 纯标准库"] --- C
    F["记忆 .brain<br/>zip + 分区 CRC + 版本校验"] --- C
```

Five layers, one rule: only synaptic weight changes. Nothing outside the
network decides what it does.

**感觉层** — Each sensor drives only a bounded patch of neurons. An unbounded
projection costs `sensors × neurons` synaptic events per step and dominates
everything else: measured, 128 sensors over 1M neurons took 836 ms/step before
this bound, 20 ms after.

**皮层** — Leaky integrate-and-fire neurons with a decaying synaptic current,
not an instantaneous kick. Without that current a spike moves its target by
1/tau_m and is gone before the next one arrives, so nothing can ever hand over
to anything — an attractor becomes impossible.

**抑制性 20%** — A separate inhibitory pool with strictly negative weights,
projecting back onto the excitatory cells. This is not decoration. Without it
the recurrent weights either do nothing or the whole network ignites on one
cue, and there is no way to sit between those.

**可塑规则** — Objects, not a pipeline. `STDP` tags by spike timing,
`Eligibility` tags by co-activity with a decay window, `Dopamine` turns a
reward into a scalar gate, `Sleep` prunes and renormalises, `Structural` grows
and removes synapses. All of them are local; none of them sees an error signal.

**海马** — Dentate expansion separates similar episodes, then CA3 stores them
in recurrent weights. Recall is a settling process: the cue is injected, then
released, and the recurrent weights decide the outcome. It can fail, which is
the difference between a memory and a lookup table.

---

## Five design rules, each learned the hard way

**1. Sparsity is a hard cap, and the rate target has to respect it.**
k-WTA allows at most `wta_k` cells to fire, i.e. `wta_k / n × 1000` Hz. Asking
homeostasis for 20 Hz while the cap sits at 1.44 Hz puts the controller in
permanent error: gain and drive climb to their ceilings and it loses all
authority. Measured, it sat at error 18.7 for 1200 steps with drive pinned and
still rising. The target is now derived from the cap.

**2. Homeostasis runs at its declared time constant.** `tau_homeostasis` was
declared and never used, while the learning rate was hand-set 17× slower than
`1/tau`. A silent network therefore needed thousands of steps to reach its
target and looked unable to self-stabilise. It now uses `1/tau`.

**3. Inhibitory weights must be strictly negative.** Drawing them from a
zero-mean normal and negating gives half of them an excitatory sign. Measured,
raising the inhibitory gain from 2.4 to 12 made the network *more* active.

**4. Inhibitory convergence must not scale with the pool.** At 4000 cells the
pool is 800 wide; a fan-in of `n_inhib/2` gave every excitatory cell 400
inhibitory synapses against 200 at 2000 cells, and completion fell from 0.99 to
0.43 — even from a *complete* cue.

**5. Scale down by functional quantity, not by ratio.** The rat CA3 has a 2%
contact probability and each episode cell receives ~60 same-episode inputs.
Copy the 2% into a 2000-cell network and each cell gets 0.8 — completion is
impossible. Keep the 60 (equivalent here to ~20) and it works at 20% density.
**The ratio was never the point; the number of same-episode inputs was.**

---

## Measured

Sparse network, NumPy backend, one process, 16-core desktop, 15.4 GB RAM.

**Throughput**

| neurons | synapses | ms/step | RSS | note |
|---:|---:|---:|---:|---|
| 10 000 | 200 000 | 0.09 | 40 MB | |
| 100 000 | 2 000 000 | 2.71 | 70 MB | |
| **139 000** | **50 000 000** | **17.1** | **1.19 GB** | fly scale, 360 synapses/cell |
| 1 000 000 | 20 000 000 | 38.2 | 366 MB | 20 synapses/cell |

**Hippocampal attractor** (4000 cells, 800 recurrent inputs, 120 cells/episode)

| condition | complete cue | half cue |
|---|---:|---:|
| CA3 alone | 100% | **100%** |
| through dentate + projection | 44% | 13% |

The second row is the current weak link and is not hidden: the projection layer
throws away half the code before CA3 sees it.

**Attractor signature** — ltp 0.10 → 0%, 0.15 → 88% (precision 100%), 0.25 →
network ignites. Cue at 10–30% → 0%, 50% → 88%, 80% → 100%. A random cue
activates **0** cells. A lookup table cannot produce that curve.

**Capacity** — 20 episodes reliable at 4000 cells, collapse at 30.

**MNIST** — 92.3% on 1000 held-out digits after one pass over 20000, against
83.3% for the same linear readout on raw pixels. The spiking features add
~9 points of separability.

**Foraging** — eligibility + dopamine raises food collected by 27% over a
non-learning control. This is the honest number for a purely local rule; an
earlier version using policy gradients scored +100% and was thrown away because
it was not a neuron.

---

### Six more rules, each learned by watching something fail

Building the organ set produced its own catalogue of mistakes. They are here
because every one of them cost a day and none of them is in a textbook.

**1. Membrane noise is not decoration.** A deterministic LIF threshold is a
step function: a cell's rate jumps from 0 to 80 Hz as its input crosses
threshold, so no homeostatic controller can hold it at a target. Every sheet
pinned its gain at the ceiling and went permanently silent. Adding noise
(σ = 0.045) turns the step into a sigmoid, and the f-I curve becomes something
a controller can work with — 2.9 Hz at zero input, 45 Hz at six.

**2. Homeostasis has to be multiplicative.** An integrating threshold
controller overshoots, saturates at its cap, and then has to unwind. Measured,
L2/3 oscillated between 5 and 60 spikes per step with a period near 600 ms,
indefinitely. A gain controller — `g *= exp(-eta * (rate/target - 1))` —
reduces error geometrically and cannot wind up.

**3. Inhibitory cells must actually inhibit.** The weight builder assigned a
positive magnitude to every synapse and used the presynaptic cell type only to
decide *which* magnitude. Result: every interneuron in every sheet was
excitatory. The network still ran, which is exactly why it took so long to
find.

**4. Afferents obey Dale's law too.** Thalamocortical weights were drawn with
random signs, so half the projection from the thalamus was inhibitory and
cancelled the input. L4 sat at zero spikes per step while L2/3 downstream was
firing: the input layer was being suppressed by its own input.

**5. Shunting inhibition divides; it does not subtract.** Implemented as
current, the relay's homeostatic gain compensated it away and the gate ran
backwards — attending a channel made it eight times louder. Implemented as a
divisor on the relay's output, a channel the cortex targets drops to 57%.

**6. An attractor needs de-meaned weights.** With `W = Σ c cᵀ`, cells that
appear in many episodes accumulate weight to everything and every cue collapses
onto them: completion was a flat 0.27 no matter how much of the cue was
present. Subtracting the mean activity before the outer product — the
covariance rule — takes the same network to 0.98.

## What does not work yet

- **Sleep replay does not consolidate.** Replaying 0, 24 or 96 episodes moves
  representation separability by 0.01 — the numbers are indistinguishable.
- **The hippocampal projection layer loses half the code.** CA3 completes 100%
  of half cues on its own codes, but only 13% once the dentate has re-encoded
  them.
- **No behaviour emerges from the snake task yet.** A non-learning brain alive
  for 60 episodes collects 0.12 food on average, and its action distribution is
  already biased (68% straight) purely from initial wiring — so "low entropy"
  is not evidence of learning.
- **Eligibility costs ~2× per step at fly scale**, because the trace array over
  50M synapses is decayed in full every step.

For the assembled brain:

- **Pattern completion stops working below ~70% of the cue.** At 70% of the
  input the cortex still reinstates the stored episode (0.97 Jaccard); at 50%
  it drops to 0.44. The entorhinal re-coding separates a degraded cue into a
  different code before CA3 ever sees it. Separation and completion pull
  against each other, and the crossover is measurable.
- **Behaviour is reflexive, not competent.** Over 8,000 steps the animal eats
  and sometimes evades, but the actions track thalamic relay and hypothalamic
  drive more than they track anything learned: cortical synapses have moved
  almost nothing in 8 seconds of simulated life.
- **The lesion suite mostly measures how much the animal moves.** Removing
  the thalamus, hypothalamus, cerebellum or neuromodulators cuts eating *and*
  being caught together (6,000 ms, one seed: intact 38 eaten / 161 caught;
  no thalamus 3 / 14; no hypothalamus 3 / 15; no neuromodulators 2 / 11), which
  says the animal became inactive rather than worse at choosing. The lesions
  are real bypasses — the unit tests check each one — but behaviour at this
  timescale cannot separate "chooses badly" from "does less".
- **The cortical motor map is arbitrary.** L5 is split into N blocks and block
  *i* means action *i*; nothing learns that correspondence. The basal ganglia
  learn the value of each channel, which is why selections are consistent, but
  the cortex cannot discover that a *different* block should drive eating.
- **High tonic dopamine disinhibits everything.** With dopamine near +1 every
  channel releases at once, which is a plausible model of the failure mode but
  is currently unregulated rather than a designed feature.

---

## Usage

No install step — this is a runnable project, not a package to depend on.
Clone it and run modules from the repository root.

```bash
# scale benchmark
python -m cerebrum.bench.scale

# tests
python -m pytest tests -q

# every ablation, one after another (takes a few minutes)
python -m cerebrum.brain.protocols

# one experiment
python -m cerebrum.brain.protocols pattern_completion
python -m cerebrum.brain.protocols lesions

# MNIST
python -m cerebrum.examples.mnist_digits --neurons 2000 --per-class 2000

# snake, with a recording you can replay in a browser
python -m cerebrum.experiments.snake_run --episodes 60 --record --out data/run.json
python -m cerebrum.experiments.make_player --record data/run.json --out player.html
```

## Modules

| file | what |
|---|---|
| `core.py` | `Brain`: assembly, sensors/actuators, rules, sleep, experience |
| `ca3.py` | spiking recurrent attractor with an inhibitory pool |
| `hippocampus.py` | dentate expansion, mossy + direct projections, episodes |
| `plasticity.py` | STDP · Eligibility · Dopamine · Sleep · Pruning · Structural |
| `memory.py` | `.brain` format: save, load, merge, graft |
| `snake.py` / `world.py` / `agent.py` | closed loops to test behaviour in |
| `backends/` | `numpy` (fast) and `pure` (standard library only) |
| `brain/sheet.py` | two-dimensional neuron sheet: positions, local E/I, divisive inhibition, multiplicative homeostasis |
| `brain/thalamus.py` | relay nucleus + reticular gate |
| `brain/cortex.py` | L4 / L2-3 / L5, self-organising map, prefrontal attractor |
| `brain/basal_ganglia.py` | direct / indirect pathways, action selection |
| `brain/hippocampus.py` | EC / DG / CA3 / CA1 with cortical reinstatement |
| `brain/amygdala.py` | one-trial fear conditioning |
| `brain/hypothalamus.py` | homeostatic drives |
| `brain/cerebellum.py` | granule / Purkinje / deep nuclei, LTD |
| `brain/neuromodulation.py` | five slow modulators |
| `brain/body.py` | grid world with food and a predator |
| `brain/whole_brain.py` | the assembly, with lesion support |
| `brain/protocols.py` | the ablation experiments |
| `lm/reservoir.py` | million-neuron spiking pool, grouped synapses, in-place update |
| `lm/model.py` | signed projection + online softmax readout |
| `lm/data.py` | character corpus, blocked interleaved split, bigram baseline |
| `lm/tasks.py` | delayed recall — the task a reservoir is actually for |
| `lm/capacity.py` | memory capacity: how many past inputs the pool holds, with a control arm |
| `lm/train.py` | the training loop |

## The `.brain` format

A ZIP of little-endian arrays plus a JSON manifest. Every part carries a
CRC32; the format version is checked on load; nothing is ever unpickled, so a
file cannot execute code.

| part | content |
|---|---|
| `manifest.json` | version, architecture signature, rule specs, scalars |
| `connectome.bin` `sensor.bin` | sparse weights, CSR |
| `neurons.bin` | membrane, threshold, adaptation, refractory, resources |
| `dentate.bin` `episodes.bin` | hippocampal codes and the events themselves |

Saved and reloaded state continues bit-identically — verified over 1000 steps
with plasticity enabled.

## License

MIT.
