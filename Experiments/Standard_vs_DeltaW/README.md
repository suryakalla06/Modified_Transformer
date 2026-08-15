# bAbI Wide-Attention vs HyperNetwork (ΔW)

This repository contains a clean, reproducible version of the two original
notebooks used to study **dynamic attention weights through a HyperNetwork**.

The experiment asks a simple question:

> Can a Transformer improve reasoning on bAbI tasks by using a small
> HyperNetwork to generate **input-dependent low-rank changes to the Q/K
> attention projections**, instead of putting the extra parameter budget
> directly into wider attention matrices?

There are **three actual model experiments**:

1. **Baseline** — fixed, widened attention.
2. **Mean → ΔW** — HyperNetwork conditioned on the masked mean of all tokens.
3. **CLS → ΔW** — HyperNetwork conditioned only on the `[CLS]` token.

The important comparison is not just accuracy: the baseline and HyperNet
variants are constructed with approximately the **same total parameter budget**,
so the experiment compares **where the capacity is placed**.

---

## 1. Repository Structure

```text
.
├── baseline.py          # Fixed wide-attention baseline
├── mean_delta_w.py      # HyperNet: sequence mean → dynamic ΔW
├── cls_delta_w.py       # HyperNet: [CLS] → dynamic ΔW
├── common.py            # Shared data, training, plotting, HyperNet components
└── README.md
```

### `baseline.py` — Wide-Attention Baseline

This is the reference model.

The Transformer keeps:

- `d_model = 128`
- `2` attention heads
- `3` attention hops/layers
- `d_ff = 256`

Instead of dynamic weights, the attention projection matrices are simply
**widened**. The resulting attention dimension is:

$$
d_{\text{attn}} = 264.
$$

Thus, the baseline spends a large fraction of its parameter budget directly on
the attention matrices:

$$
W_Q,\;W_K,\;W_V,\;W_O.
$$

The attention operation is standard scaled dot-product attention:

$$
A = \operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}}
+ M
\right),
$$

followed by

$$
\operatorname{Output}=AV.
$$

There is **no HyperNetwork and no dynamic weight update**.

---

### `mean_delta_w.py` — Mean → ΔW

This model keeps the attention width at the normal:

$$
d_{\text{model}} = d_{\text{attn}} = 128,
$$

and uses the parameter budget that would otherwise have been spent widening
attention to introduce a **HyperNetwork**.

At every attention hop, the current token representations are summarized by
a **masked sequence mean**:

$$
\bar{x}
=
\frac{\sum_{t=1}^{T} m_t x_t}
{\sum_{t=1}^{T} m_t},
$$

where

$$
m_t =
\begin{cases}
1 & \text{real token}\\
0 & \text{padding}
\end{cases}
$$

so padding does not affect the summary.

This vector is passed to the HyperNetwork:

$$
\bar{x}
\longrightarrow H
\longrightarrow
\Delta W_Q,\Delta W_K.
$$

The HyperNetwork does not generate a full dense \(128\times128\) matrix.
Instead, it produces low-rank factors with rank

$$
r=8.
$$

The resulting change is therefore of the form

$$
\Delta W \approx UV^\top,
$$

so the effective attention projections become input-dependent.

In the implementation, the resulting correction is applied to the current
normalized token representations before the attention scores are computed.
Conceptually:

$$
Q' = Q + X\Delta W_Q,
$$

$$
K' = K + X\Delta W_K.
$$

The same procedure is repeated at all **3 hops**, so each hop can generate a
different update from the current representation.

The important idea is:

> **The attention mechanism is allowed to adapt to the particular input
> sequence.**

---

### `cls_delta_w.py` — CLS → ΔW

This is almost identical to `mean_delta_w.py`.

The only change is the signal given to the HyperNetwork.

Instead of

$$
\bar{x},
$$

it uses the current `[CLS]` representation:

$$
x_{\text{CLS}} = x[:,0,:].
$$

Therefore:

$$
x_{\text{CLS}}
\longrightarrow H
\longrightarrow
\Delta W_Q,\Delta W_K.
$$

The resulting dynamic Q/K corrections are again generated at every attention
hop.

This gives the central architectural comparison:

$$
\boxed{\text{Mean}\rightarrow\Delta W
\quad\text{vs}\quad
\text{CLS}\rightarrow\Delta W}
$$

In other words:

> Should the HyperNetwork use a summary of **all tokens**, or only the
> **CLS representation**, to decide how attention should change?

---

## 2. What `common.py` Contains

`common.py` contains code that is shared so that the three experiments do not
duplicate the same implementation.

It contains:

### Data / preprocessing

- bAbI file parsing
- task loading
- tokenizer
- vocabulary construction
- padding and attention masks
- PyTorch `Dataset` / `DataLoader`

### Training

All models use the same general training procedure:

- AdamW
- cosine learning-rate schedule with warmup
- label smoothing
- gradient clipping
- early stopping
- best-checkpoint restoration

### Reproducibility

Each experiment is evaluated with the five seeds:

```text
42, 43, 44, 45, 46
```

### Evaluation / plots

The code records:

- training loss
- test accuracy
- attention heatmaps
- dynamic ΔW magnitude for the HyperNet models

### HyperNetwork components

The shared module also contains:

- `HyperNet`
- `DynamicAttentionBlock`

The two HyperNet experiments therefore differ only in **what context vector
is fed into the HyperNetwork**.

---

## 3. Experimental Protocol

The models are evaluated on six bAbI tasks:

| Task | Reasoning setting |
|---|---|
| `qa1` | single supporting fact |
| `qa2` | two supporting facts |
| `qa3` | three supporting facts |
| `qa15` | basic deduction |
| `qa16` | basic induction |
| `qa19` | path finding |

Each model is run with **five random seeds**:

$$
\{42,43,44,45,46\}.
$$

The reported result is:

$$
\text{mean accuracy} \pm \text{standard deviation}.
$$

This is important because some of the dynamic-weight improvements are strongly
seed-dependent.

---

## 4. Parameter-Budget Comparison

The comparison intentionally trades **wide fixed attention** for
**dynamic weight generation**.

For the qa1 setting, the reported parameter counts are approximately:

| Model | Parameters | Attention dimension | Main capacity allocation |
|---|---:|---:|---|
| Baseline | 642,374 | 264 | large fixed attention matrices |
| Mean → ΔW | 641,479 | 128 | HyperNet + standard attention |
| CLS → ΔW | 641,479 | 128 | HyperNet + standard attention |

For the baseline, the attention projection matrices use:

$$
128\times264
$$

per projection.

For the HyperNet models they are:

$$
128\times128,
$$

while approximately **208k parameters** are allocated to the shared
HyperNetwork.

Thus the experiment is essentially:

$$
\boxed{
\text{same overall capacity}
\;\; \text{but different allocation of that capacity}
}
$$

Baseline:

$$
\text{more fixed attention capacity}
$$

HyperNet:

$$
\text{less fixed attention capacity}
+
\text{input-dependent dynamic capacity}.
$$

---

# 5. Results

The following results are from the reported **5-seed evaluation (42–46)**.

## Mean Accuracy ± Standard Deviation

| Task | Baseline | Mean → ΔW | CLS → ΔW | Best |
|---|---:|---:|---:|---|
| `qa1` — single fact | **54.28 ± 2.46%** | **62.42 ± 9.93%** | **55.20 ± 2.41%** | Mean → ΔW |
| `qa2` — two facts | **36.04 ± 0.67%** | **35.54 ± 0.63%** | **36.04 ± 1.29%** | Baseline / CLS |
| `qa3` — three facts | **20.90 ± 1.05%** | **21.24 ± 0.80%** | **22.06 ± 0.52%** | CLS → ΔW |
| `qa15` — deduction | **70.20 ± 6.09%** | **65.24 ± 1.08%** | **74.64 ± 12.99%** | CLS → ΔW |
| `qa16` — induction | **50.18 ± 0.75%** | **49.70 ± 0.66%** | **49.58 ± 0.41%** | Baseline |
| `qa19` — path finding | **12.66 ± 0.93%** | **14.94 ± 4.07%** | **13.06 ± 2.08%** | Mean → ΔW |

---

## 6. What the Results Suggest

### `qa1` — Single Supporting Fact

Mean → ΔW improves the average result by:

$$
62.42 - 54.28 = \boxed{+8.14\text{ percentage points}}.
$$

The main reason is a strong result from seed 44:

$$
82.2\%
$$

for Mean → ΔW versus

$$
54.1\%
$$

for the baseline.

However, the standard deviation is also large:

$$
9.93\%.
$$

So the gain is promising but **not yet stable across seeds**.

---

### `qa2` — Two Supporting Facts

All three models are around:

$$
\boxed{36\%}.
$$

The dynamic-weight mechanism does not provide a meaningful improvement here.

---

### `qa3` — Three Supporting Facts

CLS → ΔW gives the strongest result:

$$
22.06\%
$$

versus

$$
20.90\%
$$

for the baseline.

The improvement is only about:

$$
\boxed{+1.16\text{ percentage points}}.
$$

So this is a relatively small effect.

---

### `qa15` — Basic Deduction

CLS → ΔW achieves:

$$
74.64\%
$$

versus

$$
70.20\%
$$

for the baseline.

The most striking run is seed 42, where CLS → ΔW reaches:

$$
\boxed{100\%}.
$$

However, the standard deviation is very high:

$$
12.99\%.
$$

Therefore this result is **interesting but highly seed-sensitive**.

---

### `qa16` — Basic Induction

All models remain close to:

$$
50\%.
$$

This suggests that the dynamic Q/K updates, at least in this configuration,
are not enough to solve this task reliably.

---

### `qa19` — Path Finding

Mean → ΔW gives:

$$
14.94\%
$$

versus

$$
12.66\%
$$

for the baseline.

This is an improvement of approximately:

$$
\boxed{+2.28\text{ percentage points}}.
$$

Again, the variance is noticeable:

$$
4.07\%.
$$

---

# 7. Main Takeaway

The experiment does **not** show that dynamic ΔW is universally better than the
baseline.

Instead, it suggests something more specific:

> **Input-conditioned low-rank changes to Q/K can help on some reasoning tasks,
> but the benefit depends strongly on the task and on how the HyperNetwork is
> conditioned.**

The clearest positive cases are:

- **Mean → ΔW on qa1**
- **CLS → ΔW on qa15**
- **Mean → ΔW on qa19**
- **CLS → ΔW gives a small edge on qa3**

But:

- `qa2` shows essentially no improvement.
- `qa16` remains around chance.
- Several of the strongest gains have **high seed variance**.

So the current evidence is best interpreted as:

$$
\boxed{
\text{Dynamic attention adaptation is promising,
but not yet uniformly reliable.}
}
$$

---

## 8. How the Three Files Relate

The clean experimental hierarchy is:

```text
                     Same Transformer task
                            │
                     Same bAbI setup
                            │
                 Same 5 random seeds
                            │
                 Similar total parameters
                            │
              ┌─────────────┴─────────────┐
              │                           │
        Fixed capacity              Dynamic capacity
              │                           │
       baseline.py                 HyperNetwork
                                      │
                              ┌───────┴───────┐
                              │               │
                        mean_delta_w.py   cls_delta_w.py
                              │               │
                         seq-mean          [CLS]
                              │               │
                              └───────┬───────┘
                                      │
                              generate ΔWQ, ΔWK
                                      │
                              modify attention
```

This is the main story of the codebase.

---

## 9. Running the Experiments

From the repository root:

```bash
python baseline.py
python mean_delta_w.py
python cls_delta_w.py
```

The default dataset path points to the Kaggle bAbI dataset used in the original
notebooks:

```text
/kaggle/input/datasets/roblexnana/
the-babi-tasks-for-nlp-qa-system/tasks_1-20_v1-2/en-10k
```

When running outside Kaggle, update `BABI_DIR` in `common.py`.

The output directory for each experiment is created under
`/kaggle/working/`, and includes training curves, attention visualizations,
and (for HyperNet models) ΔW magnitude plots.
