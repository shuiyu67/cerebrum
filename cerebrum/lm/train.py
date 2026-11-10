"""Train the reservoir language model.

    python -m cerebrum.lm.train --neurons 1000000 --tokens 12000

Prints throughput, running cross-entropy and — at the end — bits per
character against the uniform and bigram baselines. Those two baselines are
the only thing that makes the number meaningful: 2.9 bits per character is
excellent if uniform is 6.6 and embarrassing if bigram is 2.7.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .data import Corpus
from .model import ReservoirLM


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.txt")
    ap.add_argument("--neurons", type=int, default=1_000_000)
    ap.add_argument("--tokens", type=int, default=12_000,
                    help="characters to train on")
    ap.add_argument("--context", type=int, default=6)
    ap.add_argument("--k-in", type=int, default=1200)
    ap.add_argument("--bins", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=4,
                    help="milliseconds of pool dynamics per character")
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-limit", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    # Passes over the same slice. The shipped claim is one pass, online; this
    # exists to separate "the readout cannot represent it" from "one pass is
    # not enough optimisation", which the bigram comparison cannot distinguish
    # on its own — exact counting sees the whole slice, one-pass SGD does not.
    ap.add_argument("--epochs", type=int, default=1)
    # Pool shape and inhibitory rule. Exposed because the capacity measurement
    # shows they change what the pool can hold: proportional inhibition at
    # every activity level scores 1.04 against 0.71 for a threshold that only
    # engages above 3%, and fan-out changes the number of active inputs a cell
    # sees at a fixed total synapse budget.
    ap.add_argument("--fan-out", type=int, default=32)
    ap.add_argument("--w-scale", type=float, default=None,
                    help="synaptic scale; default 8/sqrt(fan_out/32)")
    ap.add_argument("--ginh", type=float, default=30.0)
    ap.add_argument("--tfrac", type=float, default=0.03,
                    help="inhibitory dead zone; 0 = proportional at all levels")
    # Log-sd of a fixed per-cell input gain. 0 keeps the old behaviour (one
    # scalar current to every driven cell, which makes the group fire in
    # lockstep); anything above 0 spreads the drive so the state reports the
    # input pattern instead of the refractory phase.
    ap.add_argument("--gain-spread", type=float, default=0.0)
    args = ap.parse_args(argv)

    corpus = Corpus(args.corpus)
    print(corpus.summary(), flush=True)

    model = ReservoirLM(
        n_neurons=args.neurons, n_chars=corpus.n_chars,
        k_context=args.context, k_in=args.k_in, n_bins=args.bins,
        steps_per_token=args.steps, seed=args.seed,
        input_gain_spread=args.gain_spread,
        fan_out=args.fan_out,
        w_scale=(args.w_scale if args.w_scale is not None
                 else 8.0 * (32.0 / args.fan_out) ** 0.5),
        ginh=args.ginh, target_frac=args.tfrac,
    )
    print(model.summary(), flush=True)

    stream = corpus.train
    start = args.context
    stop = min(len(stream) - 1, start + args.tokens)
    print(f"training on {stop - start:,} characters x {args.epochs} pass(es) "
          f"({args.steps} ms each, pool of {args.neurons:,})", flush=True)

    t0 = time.time()
    run_loss = 0.0
    run_n = 0
    pool_spikes = 0

    total = (stop - start) * args.epochs
    for i, t in enumerate(list(range(start, stop)) * args.epochs):
        ctx = stream[t - args.context:t]
        feats = model.run_pool(ctx, int(stream[t]))
        loss = model.update(feats, int(stream[t + 1]), lr=args.lr,
                            weight_decay=args.wd)
        run_loss += loss
        run_n += 1
        if run_n >= total:
            break
        pool_spikes += model.pool.spikes_total
        model.pool.spikes_total = 0

        if (i + 1) % args.eval_every == 0:
            dt = time.time() - t0
            val = model.evaluate(corpus.val, limit=args.eval_limit)
            print(f"  {i + 1:6d} chars | train {run_loss / run_n:.3f} "
                  f"| val {val['bits_per_char']:.3f} bits "
                  f"| acc {val['accuracy'] * 100:.0f}% "
                  f"| {run_n / dt:.1f} char/s "
                  f"| pool {(pool_spikes / run_n / args.steps / args.neurons * 1000):.2f} Hz",
                  flush=True)
            run_loss, run_n = 0.0, 0
            pool_spikes = 0
            t0 = time.time()

    final = model.evaluate(corpus.val, limit=1000)
    bigram = corpus.bigram_bits(train=stream[start:stop + 1])
    print("\n=== result ===", flush=True)
    print(f"  uniform      : {corpus.uniform_bits:.3f} bits/char", flush=True)
    print(f"  bigram       : {bigram:.3f} bits/char", flush=True)
    print(f"  reservoir LM : {final['bits_per_char']:.3f} bits/char "
          f"({final['accuracy'] * 100:.1f}% top-1)", flush=True)
    gain = bigram - final["bits_per_char"]
    print(f"  gain over bigram: {gain:+.3f} bits "
          f"({'better' if gain > 0 else 'WORSE — the pool is not contributing'})",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
