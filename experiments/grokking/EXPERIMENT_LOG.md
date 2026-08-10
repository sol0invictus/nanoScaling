# Grokking — Experiment & Debug Log

Chronological record of every run, measurement, bug, and decision.
Companion to [RESEARCH_PLAN.md](RESEARCH_PLAN.md). Newest entries at the bottom.

Hardware: 1× NVIDIA RTX 3090 Ti (24 GB), 16 CPU cores, WSL2, torch 2.7.1+cu126.
Reference: Power et al. 2022, *Grokking*, [arXiv:2201.02177](https://arxiv.org/abs/2201.02177).

---

## E0 — Paper setup extracted (source of truth for all hyperparameters)

Pulled from the paper's Appendix A.1.2 rather than reconstructed from memory:

| item | paper value | our setting |
|---|---|---|
| architecture | decoder-only transformer, causal mask | same |
| layers / width / heads | 2 / 128 / 4 | same |
| non-embedding params | "about 4·10⁵" | **394,496** (verified) |
| optimiser | AdamW | same |
| learning rate | 1e-3 | same |
| betas | (0.9, 0.98) | same |
| weight decay | 1.0 | same |
| warmup | linear, first 10 updates | same |
| annealing | **none** (explicit choice in the paper) | same |
| batch size | min(512, ½·|train|) | same |
| budget | 10⁵ gradient updates | see E3 |
| equation format | `<x><op><y><=><x∘y>` , loss on answer token only | 4-token prefix + logits at final position (equivalent under a causal mask, avoids unused logits) |

Paper's task list (Appendix A.1.1) — used to confirm our new task is *not* a duplicate:
8 bivariate polynomial ops mod 97 (`x+y`, `x−y`, `x/y`, `x²+y²`, `x²+xy+y²`,
`x²+xy+y²+x`, `x³+xy`, `x³+xy²+y`), one conditional op, and three ops on `S₅`.
**No dihedral group appears anywhere.**

---

## E1 — Dataset implementation & verification

Three tasks implemented in `data.py`:

| task | #symbols | #equations | role |
|---|---|---|---|
| `mod_div_97` | 97 | 9312 | positive control (the paper's Fig. 1 task) |
| `mod_add_97` | 97 | 9409 | reference; cleanest transition |
| **`dihedral_48`** | 96 | 9216 | **new task** |

Dataset sizes deliberately matched (9216 vs 9409) so dataset size is not a confound.

**Verification** (`python data.py`) — D₄₈ checked directly on the generated op table:
```
D_48: closure, identity, inverses, associativity OK; non-abelian OK
mod_add_97:  answers cover 97 distinct symbols
mod_div_97:  answers cover 97 distinct symbols
dihedral_48: answers cover 96 distinct symbols
```
Associativity tested on 2000 random triples; identity checked on both sides;
uniqueness of inverses checked per element; non-abelian confirmed (`table != table.T`).

> **Bug 1.** First run of the verification crashed with
> `NameError: name 'i' is not defined` — the inverse-uniqueness check referenced a
> loop variable that did not exist. Replaced with a vectorised
> `((table == e).sum(axis=1) == 1).all()`. Caught before any training happened.

---

## E2 — Model verification

```
non-embedding params: 394,496 (paper: ~4e5)   total: 420,352
```
Matches the paper's stated ~4·10⁵ non-embedding parameters, so we are training the
same-capacity model rather than an approximation of it.

---

## E3 — Throughput calibration (why this mattered)

First instrumented smoke run: **96 steps/s**. Suspiciously slow for a 420k-param model,
so this was investigated rather than accepted.

```
nvidia-smi:  utilization 2 %, SM clock 210 MHz (idle), power 28 W, no competing processes
```

**Diagnosis:** the model is far too small to occupy the GPU. The run is entirely
**CPU kernel-launch-bound** — the GPU never leaves idle clocks. Two consequences:

1. Optimiser launch overhead dominates → the fused AdamW kernel is a large win.
2. Concurrent runs will *not* contend for GPU compute, so the 8-run grid can be
   executed in parallel across the 16 CPU cores.

Benchmark (200 steps, batch 512, after warmup, `time.perf_counter`):

| configuration | ms/step | steps/s | warmup |
|---|---|---|---|
| eager + **unfused** AdamW | 13.81 | 72 | 0.5 s |
| eager + **fused** AdamW | **7.65** | **131** | 0.9 s |
| `torch.compile` default | 7.83 | 128 | 15.4 s |
| `torch.compile` reduce-overhead (CUDA graphs) | 2.89 | 346 | 5.7 s |

**Decision: eager + fused AdamW (131 steps/s).** CUDA graphs are 2.6× faster but
would (a) break the MLP activation capture used by the sparsity probe, which relies on
a Python-side attribute set during forward, and (b) force recompiles when the eval path
uses a different batch shape. Not worth the correctness risk inside a 2-hour budget.
Switching unfused→fused AdamW alone recovered 1.8×, which was the cheap win.

> **Bug 2.** The first benchmark printed *negative* timings (`-10.87 ms/step`).
> Cause: `time.time()` runs backwards under WSL2 (clock skew). Re-ran everything with
> `time.perf_counter()`. Worth recording because it would have silently corrupted any
> wall-clock measurement in this environment.

Measured logging overhead: 131 steps/s raw → 96 steps/s with full instrumentation
(cheap every 25 steps, expensive every 250) ≈ **27 % overhead**, accepted as the price
of resolving the transition densely.

### Cost per run

| budget | per run (alone) | 8 runs in parallel |
|---|---|---|
| 20k steps | ~3.5 min | ~4 min |
| 50k steps | ~8.5 min | ~10 min |
| 100k steps (paper budget) | ~17 min | ~20 min |

---

## E4 — First real result: D₄₈ at 50 % data does **not** grok

Smoke run, `dihedral_48`, `train_frac=0.5`, wd=1.0, 1000 steps:

```
[smoke] 0     train 4.6137/0.011  val 4.6067/0.010
[smoke] 1000  train 0.0041/1.000  val 0.0047/1.000
[smoke] DONE  t_mem=475  t_grok=500  val_acc=1.0000
```

**t_memorise = 475, t_grok = 500 → grokking gap ≈ 1.05.** Train and validation
accuracy rise essentially together: this is ordinary learning, *not* grokking.

This is consistent with the paper rather than a contradiction of it. Power et al.
Fig. 1-centre shows steps-to-generalise growing sharply as the training fraction
*decreases*, and they note that for large dataset sizes "the training and validation
curves tend to track each other more closely." D₄₈ at 50 % data sits in that easy
regime — plausibly because D₄₈ = ℤ₄₈ ⋊ ℤ₂ is "half abelian" and the transformer can
exploit the large cyclic subgroup.

**Action:** the grokking regime must be located by lowering `train_frac`. Launched a
scan at `train_frac ∈ {0.15, 0.20, 0.25, 0.30, 0.40}`, 20k steps each, light logging,
run concurrently. This scan is itself a deliverable (it is the data-efficiency curve of
the new task), not just a tuning step.

---

## E5 — Locating the grokking regime for D₄₈

Scan readout at step ≈ 4000 (`train_frac` → memorisation step and validation accuracy):

| train_frac | \|train\| | t_memorise | val acc @ 4k | regime |
|---|---|---|---|---|
| 0.15 | 1382 | 200 | 0.002 | memorised, **no** generalisation |
| 0.20 | 1843 | 300 | 0.005 | memorised, **no** generalisation |
| 0.25 | 2304 | 300 | 0.018 | memorised, **no** generalisation |
| 0.30 | 2765 | 400 | **0.737** | **mid-transition — grokking in progress** |
| 0.40 | 3686 | 700 | 1.000 | grokked at step 2100 (gap 3.0×) |

This reproduces the paper's central data-efficiency claim on a brand-new task: the
optimisation time required to generalise grows sharply as the training fraction falls,
while the memorisation time barely moves (200 → 700 steps). At 15–25 % data the network
sits in the memorising phase for the whole 4k-step window with validation accuracy at
chance (1/96 ≈ 0.0104).

**Chosen operating point for the main runs: `train_frac = 0.30`** — memorises at step
400 and generalises roughly an order of magnitude later, so the transition is wide
enough to instrument densely.

---

## E6 — Final grid launched (8 runs)

| run | task | frac | wd | seed | steps | purpose |
|---|---|---|---|---|---|---|
| R1 | `mod_div_97` | 0.5 | 1.0 | 0 | 100k | paper Fig. 1 reproduction |
| R2 | `mod_add_97` | 0.4 | 1.0 | 0 | 50k | reference; circle-embedding figure |
| R3 | `dihedral_48` | 0.30 | 1.0 | 0 | 50k | **new task, main run** |
| R4 | `dihedral_48` | 0.30 | 1.0 | 1 | 50k | seed robustness |
| R5 | `dihedral_48` | 0.30 | **0.0** | 0 | 50k | weight decay off |
| R6 | `dihedral_48` | 0.30 | **0.1** | 0 | 50k | intermediate wd |
| R7 | `dihedral_48` | 0.20 | 1.0 | 0 | 50k | less data |
| R8 | `dihedral_48` | 0.40 | 1.0 | 0 | 50k | more data |

R1 gets the paper's full 10⁵-step budget because Fig. 1 shows `x/y mod 97` at 50 % data
only reaching high validation accuracy near 10⁵ steps. R2 (`mod_add` at 40 %) is
included as a second, faster-grokking control so the reproduction does not hinge on R1
alone.

> **Bug 3 — CUDA OOM with 13 concurrent processes.** R3 and R8 died at startup with
> `RuntimeError: CUDA error: out of memory` inside `GrokkingMonitor.__init__`.
> Not a model-size problem: each process pays ~800 MB for its own **CUDA context**, and
> 13 processes (5 scan + 8 grid) initialising contexts simultaneously briefly exceeded
> 24 GB. Steady-state usage was only 9.2 GB.
> **Fix:** stagger the launches (12 s apart) instead of firing them all at once.
> Both runs relaunched successfully; all 8 now training, 9.9 GB used.
> *Lesson for the write-up:* with launch-bound micro-models the binding constraint on
> parallelism is CUDA context memory and CPU cores, not GPU compute.


---

## E7 — Optimiser study

Added an `optimizer` config field (`adamw | adam_l2 | sgd | rmsprop | muon`). Muon comes
from the repo's `optimizers/muon.py`; it orthogonalises 2-D gradients via Newton–Schulz
and 1-D parameters are handed to AdamW, wrapped in a small `Combined` optimiser. LR
warmup was changed to apply as a **multiplier on each param group's own `initial_lr`**,
so Muon's 2-D LR (0.02) and AdamW's 1-D LR (1e-3) stay in proportion.

| optimiser | lr | wd | t_mem | t_grok | gap |
|---|---|---|---|---|---|
| adamw (decoupled) | 1e-3 | 1.0 | 325 | 5750 | 17.7× |
| **muon** (+AdamW on 1-D) | 0.02 / 1e-3 | 1.0 | 300 | **425** | **1.4×** |
| adam + coupled L2 | 1e-3 | 0.1 | never | never | — |
| sgd + Nesterov | 1e-1 | 0.1 | never | never | — |
| rmsprop | 1e-3 | 0.1 | never | never | — |

**Muon essentially eliminates grokking** on this task — memorisation and generalisation
happen almost together.

> **Confound found and corrected.** The first optimiser sweep ran every optimiser at
> `weight_decay=1.0`. For AdamW that is *decoupled* decay, but for Adam/SGD/RMSprop
> PyTorch applies it as *coupled* L2 added to the gradient, which at coefficient 1.0
> destroys training outright (train accuracy stuck at chance ≈ 0.010). Re-ran the three
> coupled optimisers at wd = 0.1. They still fail to fit within budget, so the reported
> conclusion is explicitly scoped to coupled L2 at that magnitude rather than presented
> as a clean statement about the update rules.

---

## E8 — Post-grokking instability: diagnosis and fix

The first full grid produced grokking curves with **violent oscillation after the
transition** — validation accuracy repeatedly dropping from 1.0 to ~0.2 and recovering
(visible in every panel of `fig1`). The paper's curves show no such behaviour.

**Hypothesis:** weight decay 1.0 was applied to *all* parameters including LayerNorm
gains. Once the loss is near zero the gradient signal vanishes but decay does not, so the
normalisation scale is driven toward zero and the network destabilises.

**Test (R9):** identical to R3 but with 1-D parameters (LayerNorm gains, biases) excluded
from weight decay — the standard practice — via a new `wd_exclude_1d` flag.

| | R3 (decay everything) | R9 (exclude 1-D) |
|---|---|---|
| t_memorise | 325 | 300 |
| t_grok | 5750 | 4450 |
| gap | 17.7× | 14.8× |
| LayerNorm ‖W‖ trajectory | 27 → **7** (collapsing) | 27 → **60** (growing) |
| post-grok evals below 0.95 | 0.120 | **0.141** |
| post-grok minimum val acc | 0.227 | **0.013** |

> **HYPOTHESIS REJECTED — and an error of mine caught.** I first read the figure by eye
> and wrote that R9 "removed the instability entirely." Quantifying the post-transition
> window (steps > 1.2·t_grok) shows the opposite: R9 oscillates **as much or slightly
> more** than R3. The mechanism is real — the LayerNorm norm does collapse 27 → 7 when
> decayed, and grows to 60 when excluded — but it is **not** the cause of the
> oscillation. The write-up was corrected before delivery.
>
> Remaining explanation (untested, did not fit the budget): a constant learning rate with
> no annealing — which the paper explicitly chose — combined with strong decay repeatedly
> pushes the network off and back onto the general solution. An LR-annealing run would
> settle it.
>
> *Lesson: read the number, not the picture.* The oscillation envelopes in
> `fig11_stability.png` look visibly different at a glance; the statistics say they are not.

---

## E9 — Final results

Full table in `results_summary.csv`; figures in `figures/`. Headline numbers:

- **New task D₄₈ groks: gap 17.7× (seed 0), 16.6× (seed 1).**
- Weight decay: 0.0 → never generalises; 0.1 → step 31,550; 1.0 → step 5,750.
- Data fraction: 0.2 → never; 0.3 → 5,750; 0.4 → 2,100. Memorisation time barely moves.
- Internal statistics at the jump: ‖W‖ peaks at step 3,750 then falls; embedding
  effective rank 85.0 → 35.1; top-5 Fourier mass 0.29 → 0.97.

**Open discrepancy (recorded, not resolved):** R1 (`x/y mod 97`, 50% data) groks at step
725, versus ~10⁵ in the paper's Fig. 1, despite matching every stated hyperparameter.
The qualitative ordering (memorise 525 → generalise 725) holds but the magnitude does
not. This is why the main analysis is carried by D₄₈ rather than by the control.

---

## E10 — Learning-rate sweep: the optimiser conclusion in E7 was wrong

**Prompted by a reviewer question:** Muon papers use a much larger LR than AdamW, so
comparing Muon at 0.02 against AdamW at 1e-3 confounds the update rule with the step size.

**Why the LRs are not directly comparable.** Muon orthogonalises each 2-D gradient via
Newton–Schulz, so its update has ≈unit *spectral* norm; AdamW's update is element-wise
normalised, so it has ≈unit *max-abs* norm. They are different units — "both at 1e-3"
would be the unfair comparison. The principled protocol is to sweep each optimiser over
its own range and compare at each one's best.

**An implementation bug found while setting this up.** The Muon routing rule was
`p.ndim == 2`, which sent `tok_emb (98,128)`, `unembed (98,128)` and even
`pos_emb (4,128)` to Muon. Standard Muon practice (and this repo's own convention of
excluding matrices with min dim < 16) applies Muon to *hidden* weight matrices only.
This mattered specifically here because the embedding is where the structural metrics are
measured — orthogonalising its update could have manufactured the result. Added a
`muon_scope` config field (`hidden` | `all_2d`) and ran both.

**Sweep results** (dihedral_48, frac 0.3, wd 1.0):

| optimiser | lr | t_mem | t_grok | gap | stable at end |
|---|---|---|---|---|---|
| adamw | 3e-4 | 450 | none in 30k | >20× | yes |
| adamw | 1e-3 | 325 | 5750 | 17.7× | yes |
| adamw | 3e-3 | 775 | 2625 | 3.4× | yes |
| adamw | 1e-2 | 850 | 875 | **1.0×** | **no** — ends at train acc 0.33 |
| muon hidden | 5e-3 | 325 | 500 | 1.5× | yes |
| muon hidden | 2e-2 | 350 | 450 | 1.3× | yes |
| muon hidden | 5e-2 | 225 | 325 | 1.4× | yes |
| muon all_2d | 2e-2 | 300 | 425 | 1.4× | yes |

> **CONCLUSION REVISED.** E7 reported "Muon essentially eliminates grokking" and the
> write-up called this an optimiser effect. **AdamW at lr 1e-2 also reaches gap 1.0**, so
> the gap is primarily a *learning-rate* effect. My supporting argument — "Muon's
> memorisation time is unchanged, so it cannot be an effective-step-size effect" — was
> also wrong: raising the AdamW LR *increases* `t_memorise` (325 → 850) while decreasing
> `t_grok` (5750 → 875), so the two phases move in opposite directions and equal
> memorisation time does not rule out a step-size explanation.
>
> **What survives:** Muon is *insensitive* to LR (1.3–1.5× across a 10× range) whereas
> AdamW spans 17.7× → 1.0×, and AdamW's zero-gap point is unstable while Muon's is not.
> Robustness, not a lower floor.
>
> **What the scope control shows:** hidden (1.3×) vs all_2d (1.4×) — orthogonalising the
> embeddings is *not* what produces the effect. The earlier routing bug did not invalidate
> the E7 measurement, but it could have, and was only caught by asking the question.
>
> *Lesson: before attributing an effect to an optimiser, sweep the baseline's learning rate.*

---

## E11 — Transition-focused plots (fig13–fig15)

Added three views aimed specifically at "what do the statistics do when grokking emerges".

**fig15 — linear-axis zoom with per-module decomposition.** The global weight norm rises
28.0 → 34.4 (peak step 3750, just as val acc starts moving) then falls to 26.5. The
decomposition is the new information: the rise is carried almost entirely by
**attn_qkv** (10.2 → 27.1 → 16.9). `mlp_fc`, `mlp_proj`, `tok_emb` and `attn_proj` all
decline monotonically from memorisation onward and never participate in the rise. The
memorising solution lives mainly in attention.

**fig13 — transition-aligned overlay.** Rescaling each run's step axis by its own
`t_grok` collapses four runs spanning a 15× range in grokking time (2100 / 2625 / 5750 /
31550, produced by different frac, wd and lr) onto one curve for val acc, Fourier
concentration, effective rank and PCA variance. Strong evidence that the internal event
is the same regardless of when it fires. Also shows ‖W‖ peaking at ~0.85·t_grok and
`cos(∇L, W)` dipping negative exactly at t_grok.

**fig14 — event ordering, with a caveat I had to add.** First version implied several
statistics "lead" the jump. That was an artefact of the metric: "step at which half the
total change is complete" fires early for any monotonically drifting quantity (gradient
norm, ‖W‖), which is drift and not anticipation. Re-measured properly as *fraction of
total change occurring inside [0.3, 2.0]·t_grok*:

| statistic | fraction at transition |
|---|---|
| validation accuracy | 0.99 |
| Fourier concentration | 0.87 |
| embedding effective rank | 0.81 |
| top-2 PCA variance | 0.79 |

Conclusion: the structural measures are step-like and **simultaneous** with the accuracy
jump. **No clean early-warning signal was found on this task.** The figure is annotated
with the caveat so the drifting rows are not misread.

> *Third time a first-pass reading was wrong (cf. E8, E10): the plot suggested leading
> indicators, the windowed statistic said simultaneous.*

---

## E12 — Representation-theoretic analysis (irreps of D_48)

New module `irreps.py`. D_48 has 27 irreps (four 1-dim, twenty-three 2-dim;
4·1 + 23·4 = 96 = |G|), verified by Parseval on every decomposition. Statistic:
**effective number of irreps** = inverse Simpson index of the power spectrum
(27 = uniform/unstructured, 1 = single irrep).

Motivation: the Fourier measure used up to now operates on the *cyclic subgroup* and so
is blind to the non-abelian structure. 87% of the grokked embedding's power turns out to
sit in **2-dimensional** irreps — exactly what the old measure could not see.

**#1 — progress over training (fig16).** Effective #irreps is **flat at 23.4–24.3 for
the whole memorisation plateau** (steps 325 → 2000), then collapses 24 → 3.2 in lockstep
with the accuracy jump (first < 20 at step 4117; val acc > 0.05 at 3100, > 0.99 at 5750).
The `wd=0` run sits at 23.7 for its entire 40k steps.

> Confirms and strengthens the E11 negative result: **no hidden progress during the
> plateau**, even under the task-appropriate progress measure. On this task the general
> circuit is not assembled invisibly behind the memorising one.

**#4 — do different optimisers reach the same solution? (fig17)** No.

| run | t_grok | eff #irreps | dominant |
|---|---|---|---|
| adamw 1e-3 | 5750 | 3.1 | rho_21, rho_22 |
| adamw 3e-3 | 2625 | 2.8 | rho_3 |
| adamw 1e-2 | 875 | 3.3 | rho_17 |
| adamw wd0.1 | 31550 | 6.2 | rho_21, rho_7 |
| muon 0.02 | 450 | 5.0 | rho_2, rho_6 |
| muon 0.05 | 325 | 3.9 | rho_1, rho_19 |
| adamw wd0 | never | **23.7** | uniform |

All grokked runs reach 3–7 effective irreps; **which** irreps differs by optimiser and
learning rate. Universality of *form*, not of *content*. This refines §3.5: LR and
optimiser do not just change path length to a fixed solution, they select among many
equivalent sparse solutions.

> **Two bugs caught in this figure.** (a) The bar panel was y-flipped relative to the
> heatmap (`invert_yaxis` applied to one axis only), which made `wd=0` appear to have 3.1
> effective irreps instead of 23.7 — i.e. it made the *non-generalising* run look
> structured. (b) I had pre-titled the figure "converge to the SAME sparse irrep
> solution"; the data says the opposite. Both fixed before write-up.

---

## E13 — Weight-decay study (10 runs)

All D_48, frac 0.3, 30k steps, against the `t_grok=5750` baseline.

**W1 — is `lr*wd` the control variable?** Decoupled AdamW shrinks by `w <- w(1-lr*wd)`
per step, so matched products should grok together.

| lr*wd | (lr, wd) | t_grok | verdict |
|---|---|---|---|
| 1e-3 | (1e-3, 1.0) | 5750 | reference |
| 1e-3 | (3e-3, 0.333) | 6150 | holds (+7%) |
| 1e-3 | (1e-2, 0.1) | 10525 | fails (1.8x) |
| 1e-2 | (1e-2, 1.0) | 875 | reference |
| 1e-2 | (1e-3, 10.0) | never memorises | fails badly |

Approximate scaling relation near the baseline lr; breaks at the extremes. Reported with
its domain rather than as a law.

**W2 — late-onset decay.** wd on at step 2000 -> grok 7325 (5325 after onset); at 10000
-> grok 14150 (4150 after onset); baseline 5750 (5750 after onset). **The transition
clock starts when decay starts**, ~4000-5300 steps later regardless of onset. The
memorising solution is a stable attractor, and decay dislodges it on a fixed timescale
rather than instantly.

**W3 — module-selective decay.** attention-only, MLP-only and embedding-only decay
**all memorise (step 275) and never grok** in the full 30k steps. Striking because the
whole norm rise is carried by attn_qkv (E11/fig15) — decaying the module that grows is
not sufficient. Decay has to act globally.

**W4 — decay removed post-grok** (off at step 8000): val acc stays 1.000 to 30k. Decay is
needed to *reach* the solution, not to maintain it.

---

## E14 — Token-mixer ablation: Gated DeltaNet

`fla` (flash-linear-attention) is not installed, and its chunked scan (`chunk_size=64`)
is meaningless at our sequence length of **4 tokens**, so `gdn.py` re-implements the
repo's `GatedDeltaNetLayer` parameterisation exactly (same projections, gates, Q/K
sub-norms, scaling) with the recurrence unrolled. Verified: causal, depends on its own
token, gate saturates correctly; +1.66% params vs an attention block. Also patched
`metrics.py`, whose module-group mapping did not recognise GDN parameter names and would
have **silently dropped them from every norm statistic** (all 421,184 params now map).

| layers | t_mem | t_grok | eff #irreps | dominant |
|---|---|---|---|---|
| attn, attn | 325 | 5750 | 3.1 | rho_21, rho_22 |
| **attn, GDN** | 725 | **5725** | **3.2** | rho_1, rho_19, rho_2 |
| GDN, attn | 275 | 7325 | 11.4 | rho_23, sgn_s |
| GDN, GDN | 650 | **never** | **22.9** | uniform |

**Grokking is not specific to softmax attention**: one attention + one GDN groks within
0.4% of the baseline step and finds an equally sparse irrep solution (with *different*
irreps — independently reproducing the E12 "form not content" result). **Replacing both
layers kills it**: pure GDN memorises and stays at 22.9 effective irreps, the same
signature as wd=0 (23.7). One attention layer appears necessary; one is sufficient.

Discriminator now holds across optimiser, learning rate, weight decay and architecture:
**grokked => 3-11 effective irreps; not grokked => 23-24.**

*Scope:* at T=4 GDN's recurrent-state advantage is inert. This is a token-mixer ablation,
not a long-context result.

---

## Final run inventory

33 runs, all complete. `results_summary.csv` has the full table; 19 figures in `figures/`.
