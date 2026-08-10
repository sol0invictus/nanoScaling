# Grokking: Reproduction + a New Algorithmic Dataset (Dihedral Groups)

Reproduction and extension of Power et al. 2022, *Grokking: Generalization Beyond
Overfitting on Small Algorithmic Datasets* ([arXiv:2201.02177](https://arxiv.org/abs/2201.02177)).

**Budget: ~2 h wall clock, single RTX 3090 Ti.**
**Strategy: depth over breadth** — few configs, long runs, dense checkpointing, heavy
instrumentation of the transition itself.

---

## 1. Goals

1. **Reproduce** grokking faithfully on the paper's canonical task (`x/y mod 97`).
2. **Introduce a new algorithmic dataset** not in the paper — composition in the
   **dihedral group D₄₈** — and show grokking on it.
3. **Instrument the phase transition** in fine detail: weight norms, gradient
   geometry, spectral rank, logit margins, embedding structure — and identify which
   statistics *lead* the jump in test accuracy rather than merely accompanying it.

---

## 2. Datasets

Equation format follows the paper exactly: symbols are **abstract tokens with no
internal structure**, presented as `⟨x⟩⟨op⟩⟨y⟩⟨=⟩⟨x∘y⟩`. Loss and accuracy are
computed on the answer token only. Implementation feeds the 4-token prefix and
reads logits at the final position (equivalent, cheaper).

| id | definition | #symbols | #equations | role |
|---|---|---|---|---|
| `mod_div_97` | `x · y⁻¹ (mod 97)`, `y ≠ 0` | 97 | 9312 | **positive control** — the paper's Fig. 1 task |
| `mod_add_97` | `x + y (mod 97)` | 97 | 9409 | fast-grokking reference; cleanest transition to dissect |
| **`dihedral_48`** | `r^i s^f · r^j s^g` in **D₄₈** | 96 | 9216 | **NEW task** |

### The new task: dihedral group composition

`D_n` = symmetries of a regular n-gon = `{ r^i s^f : i ∈ ℤ_n, f ∈ {0,1} }`, order `2n`.
Composition rule (from `s r = r^{-1} s`):

```
(i, f) · (j, g) = ( i + (−1)^f · j  mod n ,  f ⊕ g )
```

With `n = 48` we get **96 elements and 9216 equations**, closely matched to the
paper's mod-97 tasks (9409) — so dataset size is not a confound when comparing.

**Why this task is a good addition.**
- *Genuinely new*: the paper's Appendix A.1.1 lists only modular arithmetic on ℤ₉₇
  (8 bivariate polynomial variants + a conditional task) and three tasks on the
  symmetric group `S₅`. `D₄₈` appears nowhere.
- *Non-abelian*, like `S₅`, so operand order matters — the paper notes symmetric
  operations are easier for a transformer (it can just ignore positional embeddings),
  making non-symmetric ops the more interesting regime.
- *Structurally intermediate* between the two families the paper studies: `D₄₈` is a
  semidirect product `ℤ₄₈ ⋊ ℤ₂`, i.e. a cyclic group (like mod-add) plus a single
  reflection bit. It is "half abelian", so we can ask whether the network learns the
  two factors at the **same time or at different times**.
- *Sharp structural prediction*: if the network recovers the group, the embedding of
  the 48 rotations should form a circle (as in the paper's Fig. 3-right for modular
  addition), with the 48 reflections forming a second, separate circle, and the
  reflection bit becoming a linearly decodable direction. We test all three.

---

## 3. Model & optimization (paper-faithful)

From Appendix A.1.2:
- Decoder-only transformer, causal mask: **2 layers, width 128, 4 heads**, learned
  positional embeddings, GELU MLP (4×), pre-LN. ≈ 4×10⁵ non-embedding params (verified).
- **AdamW**, lr `1e-3`, `β = (0.9, 0.98)`, **weight decay 1.0**, linear warmup over the
  first 10 updates, **no LR annealing** (the paper explicitly chose not to anneal).
- Batch size `min(512, |train|/2)`.
- Optimization budget: **10⁵ steps** (the paper's budget), subject to a throughput
  check in the smoke test; reduced only if the grid would not fit the wall clock.

---

## 4. Instrumentation — the core of the study

Logged to CSV (+ TensorBoard). **Cheap metrics every 25 steps**, **expensive every 250**.
Dense sampling is the point: the transition must be resolved, not just bracketed.

### Cheap (every 25 steps)
- train / val loss and accuracy (answer token only)
- **weight norms**: global L2, and per-module — `tok_emb`, `pos_emb`, `attn.qkv`,
  `attn.proj`, `mlp.fc`, `mlp.proj`, `unembed`
- **gradient norms**: global + per-module
- **grad–weight cosine** `cos(∇L, W)`: separates "shrink/grow the norm" from
  "rotate the solution" — the decomposition that makes the weight-decay story testable
- **update geometry**: `‖ΔW‖` per step and the cosine between consecutive updates
  (coherent descent vs. random walk)
- **logit statistics**: mean/max logit magnitude and correct-class margin, train and val
- `‖W‖ / ‖W_init‖` per module

### Expensive (every 250 steps)
- **spectral**: top-32 singular values, **stable rank** `‖W‖_F²/‖W‖₂²`, and **effective
  rank** (entropy of the normalized singular spectrum) for `W_emb`, `W_unembed`, and both MLP matrices
- **embedding geometry**: 2-D PCA of `W_emb`; variance explained by the top 2 components
- **circularity score** (task-specific progress measure): DFT power spectrum of the
  embedding rows over the cyclic index. For `mod_add_97` this is the ℤ₉₇ index; for
  `dihedral_48` we compute it **separately over the 48 rotations and the 48 reflections**.
  Report the fraction of spectral mass in the top-5 frequencies — sparse ⇒ the network
  has found a Fourier/circular representation rather than a lookup table.
- **reflection-bit decodability** (`dihedral_48` only): accuracy of a linear probe on the
  embedding predicting `f`. Tests whether the ℤ₂ factor is learned separately from ℤ₄₈.
- **neuron sparsity**: Gini coefficient of MLP activations on a fixed probe batch

### Checkpoints
Full model on a **log-spaced grid** (~40 points) for offline re-analysis, plus the
embedding matrix alone (≈50 KB) every 250 steps so embedding evolution can be
animated across the transition without storing full models.

---

## 5. Run grid (8 runs, depth-oriented)

Base = §3. Only the listed fields vary. Runs are tiny; ~8 execute concurrently on one GPU.

| run | task | train_frac | weight_decay | seed | purpose |
|---|---|---|---|---|---|
| R1 | `mod_div_97` | 0.5 | 1.0 | 0 | paper Fig. 1 reproduction |
| R2 | `mod_add_97` | 0.5 | 1.0 | 0 | cleanest transition; circle-embedding figure |
| R3 | `dihedral_48` | 0.5 | 1.0 | 0 | **new task, main run** |
| R4 | `dihedral_48` | 0.5 | 1.0 | 1 | seed robustness of the new task |
| R5 | `dihedral_48` | 0.5 | 0.0 | 0 | **wd off** — does grokking survive? |
| R6 | `dihedral_48` | 0.5 | 0.1 | 0 | intermediate wd |
| R7 | `dihedral_48` | 0.3 | 1.0 | 0 | less data ⇒ later grok (paper Fig. 1-centre) |
| R8 | `dihedral_48` | 0.7 | 1.0 | 0 | more data ⇒ earlier grok |

Weight decay and train fraction are retained because they are the two interventions the
paper identifies as decisive, and both connect directly to the recorded statistics.

---

## 6. Analyses / figures

1. **Grokking curves** — train/val accuracy vs. log steps for all three tasks (paper Fig. 1-left).
2. **Phase timing table** — `t_memorise` (train acc > 99%), `t_grok` (val acc > 99%),
   and the grokking gap `t_grok / t_memorise`, per run.
3. **Weight norm through the transition** — global and per-module ‖W‖ overlaid on val
   accuracy. Does the norm peak *before* the jump and fall as generalization arrives?
4. **Weight-decay ablation** — R3 vs R5 vs R6: steps-to-grok vs. wd, and the
   corresponding norm trajectories.
5. **Gradient geometry** — grad norm, grad–weight cosine, update coherence across the
   three phases (memorise / plateau / grok). Which statistic moves *first*?
6. **Spectral collapse** — effective and stable rank vs. step; expect a drop to a few
   effective directions at the transition.
7. **Embedding structure on the new task** — PCA of the D₄₈ embedding pre- and
   post-grok, rotations vs. reflections coloured separately; circularity score and
   reflection-bit probe accuracy vs. step.
8. **Data efficiency** — steps-to-generalize vs. train fraction (R3/R7/R8), reproducing
   the paper's Fig. 1-centre trend on the new task.

---

## 7. Deliverables

```
experiments/grokking/
├── RESEARCH_PLAN.md      # this file
├── data.py               # task generators: modular arithmetic, dihedral group
├── model.py              # 2L / d128 / 4H decoder-only transformer
├── metrics.py            # GrokkingMonitor — all internal statistics
├── train.py              # single-run loop: CSV + TensorBoard + checkpoints
├── configs/*.yaml        # the 8 run configs
├── run_grid.sh           # concurrent launcher
├── analysis.ipynb        # all figures, reads the CSVs
├── figures/
└── WRITEUP.md            # short report: setup, results, interpretation
```
Plus the full prompt / AI-conversation transcript exported alongside.

---

## 8. Schedule

| time | activity |
|---|---|
| 0:00–0:20 | build `data.py` / `model.py` / `metrics.py` / `train.py`; unit-check the D₄₈ group law (associativity, identity, inverses) and param count |
| 0:20–0:30 | smoke test: 500 steps, measure steps/s, **fix the step budget** |
| 0:30–0:35 | launch all 8 runs concurrently |
| 0:35–1:20 | runs execute; build the analysis notebook against partial CSVs meanwhile |
| 1:20–1:40 | final figures |
| 1:40–2:00 | `WRITEUP.md` + transcript export |

**Risk controls**
- `mod_div_97` is a known-grokking positive control: it guarantees a reproduction
  deliverable even if `dihedral_48` behaves unexpectedly.
- CSVs are line-buffered and append-mode, so the notebook can be developed against
  partial results while runs are still going.
- Log-spaced checkpoints mean any analysis can be redone offline without retraining.
- If throughput is worse than expected, the step budget is cut uniformly across runs
  (recorded in the write-up) rather than dropping runs.
