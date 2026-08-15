# Dynamic Attention Through Positional and HyperNetwork Weight Modifications

This repository studies a central question about Transformer attention:

> **Can modifying the Query and Key projections in a structured,
> input-dependent way improve reasoning performance?**

The work explores this question in two stages.

---

# 1. Overall Idea

A standard Transformer learns fixed attention projection matrices:

$$
Q = XW_Q,
\qquad
K = XW_K.
$$

Attention is then computed as

$$
A =
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_h}} + M
\right).
$$

The research idea is to introduce an additional modification:

$$
W_Q \rightarrow W_Q + \Delta W_Q,
$$

$$
W_K \rightarrow W_K + \Delta W_K.
$$

This changes the effective attention mechanism without replacing the entire
Transformer.

The project investigates two different ways of obtaining this extra
information.

---

# 2. Experiment 1 — Positional ΔW

### Folder

**Standard vs PosDelta**

This experiment asks:

> **What happens if the attention mechanism receives a structured positional
> modification?**

The Q/K projections become approximately

$$
Q' = Q + \alpha_Q X\Delta W_Q
$$

and

$$
K' = K + \alpha_K X\Delta W_K.
$$

The positional \(\Delta W\) tensors are constructed from sinusoidal positional
features using a rank-4 block structure.

The matrices themselves are fixed.

Only the scalar strengths

$$
\alpha_Q,\alpha_K
$$

are learned.

Thus this experiment introduces **structured positional information into the
attention weights**, but does not make the modification input-dependent.

### What this experiment tells us

It tests whether there is value in giving attention an explicit positional
correction at the Q/K level.

The final results are:

| Task | Standard | PosDelta | Δ |
|---|---:|---:|---:|
| `bAbI-qa1` | 55.60 ± 2.82% | 52.70 ± 1.36% | −2.90 pp |
| `bAbI-qa2` | 34.96 ± 0.56% | 35.80 ± 0.36% | +0.84 pp |
| `bAbI-qa3` | 22.16 ± 1.38% | 21.40 ± 0.92% | −0.76 pp |
| `bAbI-qa15` | 71.06 ± 5.85% | 61.22 ± 0.93% | −9.84 pp |
| `bAbI-qa16` | 49.62 ± 0.55% | 49.96 ± 0.59% | +0.34 pp |
| `bAbI-qa19` | 13.44 ± 2.17% | 21.90 ± 5.49% | +8.46 pp |

The main observation is that the positional modification is **task-dependent**:
it helps substantially on path finding (`qa19`) but hurts substantially on
basic deduction (`qa15`).

---

# 3. Experiment 2 — HyperNetwork ΔW

### Folder

**Standard vs ΔW / HyperNet**

The second experiment asks a more general question:

> **Instead of fixing ΔW, can a HyperNetwork generate an input-dependent ΔW?**

The model keeps the standard attention dimension:

$$
d_{\text{model}} = 128.
$$

A HyperNetwork receives a representation of the current input and produces
low-rank corrections for Q and K:

$$
h = H(c),
$$

$$
\Delta W_Q = U_QV_Q^T,
\qquad
\Delta W_K = U_KV_K^T,
$$

with low rank

$$
r=8.
$$

The effective projections become

$$
Q' = Q + X\Delta W_Q,
$$

$$
K' = K + X\Delta W_K.
$$

The crucial difference is that

$$
\Delta W_Q,\Delta W_K
$$

are now **input-dependent**.

---

# 4. Two HyperNetwork Variants

The HyperNetwork experiment compares two conditioning signals.

## Mean → ΔW

The HyperNetwork receives a masked mean of the current sequence:

$$
c =
\frac{\sum_t m_t x_t}
{\sum_t m_t}.
$$

Then:

$$
c
\rightarrow H
\rightarrow
\Delta W_Q,\Delta W_K.
$$

This gives the HyperNetwork a summary of the entire input sequence.

---

## CLS → ΔW

The HyperNetwork instead receives the `[CLS]` representation:

$$
c = x_{\text{CLS}}.
$$

Then:

$$
x_{\text{CLS}}
\rightarrow H
\rightarrow
\Delta W_Q,\Delta W_K.
$$

This tests whether the learned CLS representation is a better conditioning
signal than the masked mean.

---

# 5. Why There Are Two Different Experiments

The project is therefore exploring **two increasingly powerful forms of
attention modification**.

```text
                    Standard Transformer
                           |
                           v
                 Can attention benefit
                  from ΔW modifications?
                           |
              ┌────────────┴────────────┐
              │                         │
              v                         v
       Positional ΔW              HyperNet ΔW
       fixed structure             dynamic structure
              │                         │
              │                  ┌──────┴──────┐
              │                  │             │
              │               Mean → ΔW    CLS → ΔW
              │
              v
      Does explicit positional
       structure help attention?
```

So:

### Experiment 1

$$
\boxed{
\text{Fixed positional structure}
\rightarrow
\Delta W
}
$$

### Experiment 2

$$
\boxed{
\text{Input representation}
\rightarrow
\text{HyperNetwork}
\rightarrow
\Delta W
}
$$

The second experiment is therefore a **dynamic/generalized version of the
first idea**.

---

# 6. Common Evaluation Setup

Both experiments use the bAbI reasoning benchmark.

The evaluated tasks are:

| Task | Reasoning problem |
|---|---|
| `qa1` | Single supporting fact |
| `qa2` | Two supporting facts |
| `qa3` | Three supporting facts |
| `qa15` | Basic deduction |
| `qa16` | Basic induction |
| `qa19` | Path finding |

The models are evaluated over five random seeds:

$$
42,43,44,45,46.
$$

The main metric is:

$$
\text{mean accuracy} \pm \text{standard deviation}.
$$

This multi-seed evaluation is important because some of the improvements from
dynamic ΔW are sensitive to initialization.

---

# 7. What the Project Is Trying to Understand

The broader research question is not simply:

> "Can I get higher accuracy?"

It is:

> **Can the attention mechanism itself be adapted to the structure of the
> current reasoning problem by modifying its Q/K projections?**

There are several progressively more flexible possibilities:

### Fixed standard attention

$$
W_Q,\;W_K
$$

are learned once and remain fixed for every example.

### Positional ΔW

$$
W_Q + \alpha_Q\Delta W_Q,
\qquad
W_K + \alpha_K\Delta W_K
$$

introduce a fixed positional structure.

### HyperNetwork ΔW

$$
W_Q + \Delta W_Q(x),
\qquad
W_K + \Delta W_K(x)
$$

allow the attention mechanism to change according to the current input.

This gives the project a natural progression:

$$
\boxed{
\text{Fixed Attention}
\rightarrow
\text{Structured Positional Attention}
\rightarrow
\text{Input-Conditioned Attention}
}
$$

---

# 8. Current Findings

The PosDelta experiment shows that fixed positional modifications are not
universally beneficial.

They produce:

- a strong positive result on `qa19`
- small improvements on `qa2` and `qa16`
- degradation on `qa1`, `qa3`, and especially `qa15`

The HyperNetwork experiment provides evidence that **input-conditioned ΔW can
produce larger task-specific changes**, including improvements on some
reasoning tasks, but those gains can be seed-sensitive.

This means the interesting question moving forward is not simply whether ΔW
exists, but:

> **What information should control ΔW, and for which reasoning structures is
> dynamic attention actually useful?**

That is the role of the Mean → ΔW and CLS → ΔW comparison.

---

# 9. Repository Organization

The project can therefore be organized as two experimental folders:

```text
.
├── standard_vs_posdelta/
│   ├── standard_vs_posdelta.py
│   ├── data_utils.py
│   └── README.md
│
└── standard_vs_hypernet/
    ├── baseline.py
    ├── mean_delta_w.py
    ├── cls_delta_w.py
    ├── common.py
    └── README.md
```

The first folder answers:

> **Does a structured positional ΔW help?**

The second folder answers:

> **Can a HyperNetwork generate useful input-dependent ΔW, and should it be
> conditioned on the sequence mean or on `[CLS]`?**

Together, these experiments form a study of **adaptive attention through
structured modifications of the Q/K projections**.
