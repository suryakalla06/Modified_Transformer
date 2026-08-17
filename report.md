# Adaptive Transformer Attention — Engineering Report

An empirical study of **input-conditioned attention weights** in small transformers,
evaluated on the bAbI reasoning tasks under a matched parameter budget.

This report explains the design of the study and what it found. The README covers
how to run it; this covers why it is built the way it is, and — as much as the
results themselves — what the results are *not* strong enough to claim.

---

## 1. Problem and scope

A standard transformer learns one set of projection matrices `W_Q`, `W_K` and
applies them unchanged to every input. The question here is narrow and testable:

> If those projections are allowed to shift **per input**, does a small
> transformer get better at multi-step reasoning — or does it just get bigger?

That second clause is the whole difficulty. Any mechanism that generates weights
adds parameters, and on a small model added parameters alone can move accuracy.
A comparison that does not control for this measures capacity and calls it
architecture. Everything below is shaped by controlling for it.

**In scope:** two families of adaptive-attention variants, six bAbI tasks, five
seeds each, against parameter-matched baselines.

**Out of scope, deliberately:** large-scale pretraining, natural-language
benchmarks, and inference-cost measurement. This is a controlled comparison of
architectures at small scale, not a claim that any of it survives at scale. There
is no throughput or latency benchmark in this repository and therefore no
performance claim anywhere in this report.

---

## 2. Architecture and data flow

Two independent experiments live under `Experiments/`, each with its own baseline.

### 2.1 `Standard_vs_DeltaW` — HyperNetwork-generated ΔW

Three models, one shared training harness:

| Model | Where the extra capacity goes |
|---|---|
| **Baseline** | a *wider* fixed attention matrix (`d_attn = 264`) |
| **Mean → ΔW** | a HyperNetwork conditioned on the masked-mean sequence representation |
| **CLS → ΔW** | a HyperNetwork conditioned on the `[CLS]` representation |

Data flow for a ΔW variant, per hop:

```
token ids ──► embedding + learned positional ──► LayerNorm ──► x
                                                               │
              ┌────────────────────────────────────────────────┤
              │                                                │
    summary vector v                                    Q,K,V = W·LN(x)
    (x[:,0,:] for CLS,                                         │
     masked mean for Mean)                                     │
              │                                                │
        HyperNet(v) ──► U,V factors ──► ΔQ = xVUᵀ ────────────►│  Q += ΔQ
                                        ΔK = xVUᵀ ────────────►│  K += ΔK
                                                               ▼
                                             softmax(QKᵀ/√d_h + pad_mask)·V
                                                               │
                                              residual ──► FFN ──► x'
```

The three hops each call the **same** HyperNet instance, but on that hop's own
`x` — so the deltas are recomputed as the representation evolves, not fixed once
at the input.

`ΔW` is never materialised as a `d × d` matrix. `HyperNet` emits two low-rank
factors and the block applies them as `x → (x·V)·Uᵀ`, keeping the cost linear in
`rank` rather than quadratic in `d_model`. With `rank = 8` against `d_model = 128`
that is the difference between the idea being testable at this budget and not.

### 2.2 `standard_vs_posDelta` — fixed positional ΔW

A cheaper variant of the same idea, and a deliberate ablation of it: instead of
*generating* the delta from content, apply a **fixed, learned, rank-k positional**
ΔW to Q/K, scaled by two learnable scalar gates. Two differences from the
Standard model, and both matter for reading the result:

- **the ΔW itself** — positional and learned-once, not content-generated
- **the readout** — Standard uses a `[CLS]` token; PosDelta uses a masked mean

Because two things change at once, this experiment cannot attribute its
differences to the ΔW alone. That is a real weakness of the design and is stated
again in §6 rather than buried.

---

## 3. Module walkthrough

| File | What it owns |
|---|---|
| `Experiments/Standard_vs_DeltaW/common.py` | Everything shared: `Config`, tokenizer, bAbI parser, loaders, the training loop, plotting, and both `HyperNet` and `DynamicAttentionBlock`. |
| `…/baseline.py` | `WideAttentionBlock` + `StandardWideBaseline`, including the parameter-matching arithmetic (§4.1). |
| `…/mean_delta_w.py` | `MeanDeltaW` — conditions the HyperNet on `seq_mean(x, mask)`. |
| `…/cls_delta_w.py` | `CLSDeltaW` — conditions the HyperNet on `x[:, 0, :]`. |
| `Experiments/standard_vs_posDelta/standard_vs_posdelta.py` | Both models of the second experiment, self-contained. |
| `…/data_utils.py` | Data handling for that experiment. |
| `analysis/seed_variance.py` | Post-hoc paired bootstrap over seeds (§4.4). |

Keeping one `common.py` is the reason the three-way comparison is trustworthy:
the models differ **only** in their attention block and their conditioning
vector. Tokenisation, vocabulary cap, batching order, optimiser, schedule, early
stopping and evaluation are one implementation, used by all three. Had each
script carried its own loop, any difference in the results could equally have
been a difference in the harness.

The shared training loop (`train_model`) fixes for every model: AdamW at
`lr = 5e-4`, `weight_decay = 1e-2`, a cosine schedule with 10% warmup, gradient
clipping at 1.0, cross-entropy with `label_smoothing = 0.1`, 60 epochs with
early stopping at `patience = 20`, and **restoration of the best checkpoint** —
so the reported number is best test accuracy, identically defined across models.

---

## 4. Design decisions

### 4.1 Matching the parameter budget — the decision the study rests on

The baseline is not a plain transformer. It is a transformer **widened until it
costs what the HyperNetwork costs**. `baseline.py` computes the HyperNet's
parameter count analytically, divides it across the three hops, converts that
into extra attention width (`4 · d_model` parameters per unit of width, for
Q/K/V/O), and rounds up to a multiple of `n_heads`:

```python
extra_params_per_hop = hyper_params // cfg.n_hops
extra_d              = extra_params_per_hop // (4 * cfg.d_model)
d_attn               = cfg.d_model + extra_d          # → 264
```

| Model | Parameters | Attention dim |
|---|---:|---:|
| Baseline | 642,374 | 264 |
| Mean → ΔW | 641,479 | 128 |
| CLS → ΔW | 641,479 | 128 |

**0.14% apart.** Both models are given the same budget and differ only in how
they spend it — the baseline on larger fixed matrices, the ΔW variants on a
generator for small ones. Without this, every number in §5 would be unreadable.

### 4.2 Low-rank factors instead of a generated matrix

Generating a full `d × d` delta per head per hop would dominate the parameter
count and defeat 4.1 before the experiment started. `HyperNet` instead emits `U`
(`n_heads × d_head × rank`) and `V` (`n_heads × d_model × rank`) for Q and K, and
the block applies them as two matmuls. `rank = 8` and `hyper_hidden = 32` were
chosen to keep the generator small enough that the matched baseline stays a
sensible model rather than an absurdly wide one.

The generator is `Linear → LayerNorm → GELU → Linear → GELU → Linear`, with a
single learnable `scale` parameter applied as `.abs()` so the delta magnitude can
be damped toward zero during training but never sign-flipped wholesale.

### 4.3 Five seeds, six tasks — before looking at any result

`SEEDS = [42, 43, 44, 45, 46]` is fixed in `common.py`, and every model runs all
five on all six tasks. `seed_everything` sets Python, NumPy and Torch RNGs and
forces `cudnn.deterministic`; the `DataLoader` gets its own seeded generator so
shuffling order is reproducible too.

The six tasks were picked to span reasoning depth rather than to be easy:
single-fact (`qa1`), two-fact (`qa2`), three-fact (`qa3`), deduction (`qa15`),
induction (`qa16`), path-finding (`qa19`).

### 4.4 Reporting the spread, and then distrusting it

This is the decision I would most want to defend in a review.

The single largest effect in the study is **Mean → ΔW on qa1: +8.14 pp**. Its
standard deviation is **9.93 pp**. Reported as a bare mean it is a result;
reported with its spread it is not yet distinguishable from the choice of random
seed. The same holds for CLS → ΔW on qa15: **+4.44 pp** with **sd = 12.99 pp**.

`analysis/seed_variance.py` makes that reading mechanical rather than a matter of
eyeballing a table. For every (task, variant-pair) it computes a **paired
bootstrap** interval — 10,000 resamples, 95% interval, its own RNG fixed at seed 0
so the tool is not itself a source of variation — and labels a comparison
`SEPARATED` only when the interval excludes zero.

It resamples **seeds, not runs**, because both variants are evaluated on the same
seed set. A seed that happens to be easy inflates both; preserving that pairing is
the entire point, and resampling variants independently would widen the interval
for no reason.

The script prints a caveat with its own output, and it belongs here too: with
n = 5 the interval is itself estimated from five numbers. It is a guard against
over-claiming, **not** a significance test. `not separated` means the experiment
cannot tell the two apart — not that they are equal.

### 4.5 Making the results reproducible off Kaggle

The experiments were first run on Kaggle, where the bAbI corpus sits at a fixed
input path. Hardcoding it meant the code could not run — and the results could not
be checked — anywhere else. Both paths are now environment-overridable with the
original Kaggle location as the fallback, so existing notebooks keep working:

```bash
export BABI_DIR=/path/to/tasks_1-20_v1-2/en-10k
export OUT_ROOT=./runs
```

`requirements.txt` is pinned to the versions the reported numbers were produced
under, with a note that they should only be loosened by someone willing to re-run
the full 5-seed sweep and confirm the numbers hold.

### 4.6 Instrumentation kept in the harness

`common.py` emits, for seed 42 only, per-task training curves, attention heatmaps
across all three hops, and — for the ΔW variants — the **Frobenius norm of the
generated ΔQ per epoch**. That last one is the diagnostic that distinguishes a
mechanism that is doing something from one the optimiser has quietly zeroed: if
the delta magnitude collapses, the variant has converged to its own baseline
regardless of what the accuracy column says.

---

## 5. Results

### 5.1 Wide attention vs generated ΔW — mean ± sd over 5 seeds

| Task | Baseline | Mean → ΔW | CLS → ΔW | Best |
|---|---:|---:|---:|---|
| `qa1` — single fact | 54.28 ± 2.46% | **62.42 ± 9.93%** | 55.20 ± 2.41% | Mean → ΔW |
| `qa2` — two facts | **36.04 ± 0.67%** | 35.54 ± 0.63% | **36.04 ± 1.29%** | Baseline / CLS |
| `qa3` — three facts | 20.90 ± 1.05% | 21.24 ± 0.80% | **22.06 ± 0.52%** | CLS → ΔW |
| `qa15` — deduction | 70.20 ± 6.09% | 65.24 ± 1.08% | **74.64 ± 12.99%** | CLS → ΔW |
| `qa16` — induction | **50.18 ± 0.75%** | 49.70 ± 0.66% | 49.58 ± 0.41% | Baseline |
| `qa19` — path finding | 12.66 ± 0.93% | **14.94 ± 4.07%** | 13.06 ± 2.08% | Mean → ΔW |

### 5.2 Standard vs fixed positional ΔW

| Task | Standard | PosDelta | Δ |
|---|---:|---:|---:|
| `bAbI-qa1` | **55.60 ± 2.82%** | 52.70 ± 1.36% | −2.90 pp |
| `bAbI-qa2` | 34.96 ± 0.56% | **35.80 ± 0.36%** | +0.84 pp |
| `bAbI-qa3` | **22.16 ± 1.38%** | 21.40 ± 0.92% | −0.76 pp |
| `bAbI-qa15` | **71.06 ± 5.85%** | 61.22 ± 0.93% | −9.84 pp |
| `bAbI-qa16` | 49.62 ± 0.55% | **49.96 ± 0.59%** | +0.34 pp |
| `bAbI-qa19` | 13.44 ± 2.17% | **21.90 ± 5.49%** | +8.46 pp |

### 5.3 Reading them honestly

Three things are worth saying, in this order.

**The gains are not uniform, and that is the informative part.** Where adaptive
attention helps it helps on the tasks with the least structure to memorise
(`qa1`, `qa19`); where the baseline wins it wins on tasks where a larger fixed
projection is simply the better use of the budget (`qa16`, `qa2`). A mechanism
that helped everywhere by a small margin would be a capacity effect that survived
§4.1 by accident.

**The largest gains carry the largest variance.** `qa1` at +8.14 pp comes with
sd = 9.93 pp; `qa15` at +4.44 pp with sd = 12.99 pp. The tasks with the tightest
spread (`qa2`, `qa16`, sd < 1 pp) are the ones where nothing much happened. This
is exactly the pattern you would expect if the big numbers were partly seed luck,
and it is why §4.4 exists.

**Changing two things at once limits §5.2.** PosDelta differs from Standard in
both its ΔW and its readout, so its −9.84 pp on `qa15` and +8.46 pp on `qa19`
cannot be attributed to the ΔW alone.

The defensible summary: **input-conditioned attention shifts where a small
transformer spends its capacity, with effects that are task-dependent and, at
n = 5 seeds, mostly not separable from seed noise.** Anything stronger than that
is not supported by what is in this repository.

---

## 6. Limitations

1. **n = 5 seeds.** Enough to expose variance, not enough to establish
   significance. The bootstrap intervals are estimated from five numbers.
2. **No committed results CSV.** `analysis/seed_variance.py` consumes the
   per-seed CSV that the experiment writes, so reproducing the verdicts requires
   re-running the sweep. Committing the CSVs would make the analysis checkable
   without a GPU, and is the single cheapest improvement available.
3. **§5.2 confounds two changes** — the ΔW and the CLS→mean readout switch.
   Splitting it into two ablations would make it interpretable.
4. **Small scale, synthetic data.** `d_model = 128`, 2 heads, 3 hops, ~640k
   parameters, vocabulary capped at 200 for bAbI. bAbI is templated text; nothing
   here transfers to natural language without being re-run.
5. **Absolute accuracies are low** on the harder tasks (`qa3` ~21%, `qa19`
   ~13%). These models are near chance on multi-hop reasoning, so the comparison
   is between two weak models — a real caveat on how much the ranking means.
6. **Cost is unmeasured.** The HyperNetwork runs once per hop per forward pass
   and is certainly not free, but there is no timing benchmark here, so no
   efficiency claim is made either way.

## 7. What I would do next

- Commit the per-seed CSVs so `seed_variance.py` runs without a GPU.
- Extend to 15–20 seeds on `qa1` and `qa15` alone — the two tasks where the
  effect is large enough to be worth resolving — rather than more tasks at n = 5.
- Split the PosDelta experiment into two clean ablations.
- Log ΔW magnitude for every seed, not just seed 42, and check whether the
  high-variance runs are the ones where the delta stayed large.

---

## 8. Reproduction

```bash
pip install -r requirements.txt

export BABI_DIR=/path/to/tasks_1-20_v1-2/en-10k    # Kaggle path is the default
export OUT_ROOT=./runs

cd Experiments/Standard_vs_DeltaW
python baseline.py        # wide-attention control
python mean_delta_w.py    # HyperNet conditioned on the masked mean
python cls_delta_w.py     # HyperNet conditioned on [CLS]

cd ../standard_vs_posDelta
python standard_vs_posdelta.py

python ../../analysis/seed_variance.py runs/standard_vs_posdelta/comparison_results.csv
```

Each script runs all six tasks across all five seeds and prints a per-task
mean ± sd. Plots for seed 42 land under `$OUT_ROOT`. Expect a GPU: this is
5 seeds × 6 tasks × up to 60 epochs per script.
