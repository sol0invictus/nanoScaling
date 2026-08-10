# Grokking on the dihedral group D₄₈

A mechanistic study of grokking: reproducing Power et al. (2022,
[arXiv:2201.02177](https://arxiv.org/abs/2201.02177)) and extending it to a new algebraic
dataset. 33 runs, 8 axes, 19 figures.

**Headline:** on a task the paper never ran, the network learns its 2765 training
examples by step 325 while still scoring at chance on held-out ones, then jumps to perfect
held-out accuracy at step 5750. Generalisation takes 17.7 times as long as memorisation.

What changes at that moment is measurable. Group theory gives D₄₈ a fixed set of 27
irreducible building blocks, the natural coordinate system for anything respecting the
group's structure. While memorising, the network spreads its representation almost evenly
across all 27, which is what an unstructured lookup table looks like. At the jump it
collapses onto about 3, and the two dominant ones turn out to be exactly enough to tell all
96 group elements apart. The network compresses to the minimum structure that still
represents the group faithfully, and stops.

That count separates every run that generalises (between 3 and 11 blocks in use) from every
run that never does (23 to 24), across optimisers, learning rates, weight decays and architectures. It is the
best single predictor we found, and it never moves before the jump.

---

## Where to start

**If you have five minutes** and want the findings, read the headline bullets at the top of
`REPORT.md`. They are written for someone who has not seen the rest, and define the terms
they use.

**If you have longer**, read the whole of `REPORT.md`. Each results subsection is laid out
as hypothesis, test, finding, and what it implies for the overall picture, and every figure
is inlined.

**If you want to check a specific number**, open `results_summary.csv`. One row per run
with the config, the memorisation step, the grokking step and the gap.

**If you want to see the raw curves**, open `analysis.ipynb`, or just browse `figures/`.

**If you care how the study was designed** rather than what it found, read §4 of
`REPORT.md`. It covers how an open-ended brief was narrowed to eight axes, what was
deliberately left out, and how the runs were instrumented so that later questions could be
answered without retraining. The single best example: the irrep analysis that produced the
central result was written after every run had finished and cost no additional GPU time,
because full checkpoints were saved on a log-spaced grid.

**If you want to know what we are unsure about**, read §8 of `REPORT.md`, which separates
open questions (things we would test next, and why they matter) from limitations on what we
actually claim.

**If you are reviewing the work rather than the results**, read `EXPERIMENT_LOG.md`. It is
the chronological record, including the three conclusions that were wrong until measured
and the bugs that produced them. §6 of `REPORT.md` is the short version. These are the most
honest documents here.

**If you want to run it**, jump to *Reproducing* below.

---

## The files

### Read these

| File | What it is |
|---|---|
| `REPORT.md` | **The report.** Findings, figures, conclusions. Start here. |
| `EXPERIMENT_LOG.md` | Chronological lab log: every run, every measurement, every bug, and the reversed conclusions. |
| `RESEARCH_PLAN.md` | The plan written before any training, kept unedited so the design can be compared against what actually happened. |
| `results_summary.csv` | One row per run: config, t_memorise, t_grok, gap, final accuracies. |
| `analysis.ipynb` | The figures, interactively, regenerated from the CSV logs. |
| `report.html` | `REPORT.md` as a standalone page with all figures embedded. Built from `report_template.html` by inlining the PNGs. |
| `WRITEUP.md` | Superseded draft of the report, kept only as a record of the intermediate state. Read `REPORT.md` instead. |

### The code, in dependency order

| File | Lines | What it does |
|---|---|---|
| `data.py` | 203 | The three tasks: `dihedral_48` (new), `mod_div_97` and `mod_add_97` (controls). Running it verifies the D₄₈ group axioms directly on the generated table. |
| `model.py` | 126 | The paper's 2-layer, width-128, 4-head decoder-only transformer. `layer_types` selects the token mixer per layer. |
| `gdn.py` | 117 | Gated DeltaNet layer, recurrence unrolled. Self-contained; does not need `flash-linear-attention`. Running it checks causality. |
| `muon.py` | 82 | The Muon optimiser, vendored so this directory has no parent-repo dependency. |
| `metrics.py` | 282 | `GrokkingMonitor`: all the internal statistics, written to CSV as training runs. |
| `irreps.py` | 121 | Decomposition into the 27 irreducible representations of D₄₈. This is where the mechanistic result comes from. |
| `train.py` | 327 | One run. Config via YAML plus `--key=value` overrides. |
| `analysis.py` | 936 | Every figure. `python analysis.py` regenerates all 19. |

### Generated

| Path | What it is |
|---|---|
| `configs/` | 33 run configs, one YAML each, named by study (`r*` main, `lr_*` learning rate, `wd_*` weight decay, `o_*` optimiser, `g_*` token mixer). |
| `runs/` | Per-run CSV logs, checkpoints on a log-spaced grid, and `metrics.json`. Not in version control. |
| `figures/` | The 19 figures. |
| `report_template.html` | Source for `report.html`, with `{{FIG:...}}` placeholders instead of inlined images. |

---

## Self-contained

This directory does not import from the parent repository. Verified by copying the code
into an empty directory with `PYTHONPATH=` and running every entry point there.

Beyond the Python standard library it needs only:

```
torch  numpy  pandas  matplotlib  pyyaml
```

Muon is vendored as `muon.py` (byte-identical to the original), and `gdn.py` reimplements
the Gated DeltaNet recurrence rather than depending on `flash-linear-attention`, which is
not installed and whose chunked kernel would be meaningless at our sequence length of 4
tokens anyway.

---

## Reproducing

Four checks that need no training and take a few seconds each:

```bash
python data.py     # D_48 closure, identity, inverses, associativity, non-commutativity
python model.py    # parameter count: 394,496 non-embedding, matching the paper's ~4e5
python gdn.py      # the GDN layer is causal, and iso-parameter with attention
python irreps.py   # 27 irreps of D_48, dimensions summing to |G| = 96
```

Then the runs. A single run is a few minutes; the full grid is a few hours of elapsed time
on one GPU because the runs execute concurrently.

```bash
python train.py configs/r3_dihedral_main.yaml   # the headline run on its own
bash run_grid.sh                                # everything
python analysis.py                              # regenerate all figures
```

One caveat on the grid: stagger the launches. The model is far too small to occupy a GPU
(ours sat at 2% utilisation), so the runs are CPU launch-bound and parallelise well, but
each process pays about 800 MB for its CUDA context. Starting a dozen at the same instant
will exhaust a 24 GB card during initialisation. `run_grid.sh` does this for you.

Overriding a config from the command line:

```bash
python train.py configs/r3_dihedral_main.yaml --weight_decay=0.0 --max_steps=50000
python train.py --task=dihedral_48 --train_frac=0.3 --optimizer=muon --muon_lr=0.02
python train.py --task=dihedral_48 --layer_types=attn,gdn
```

---

## Hardware this was run on

One NVIDIA RTX 3090 Ti (24 GB), 16 CPU cores, PyTorch 2.7.1 / CUDA 12.6. Roughly 1.0M
optimisation steps across 40 runs, about 10 hours of summed run time compressed into a few
hours elapsed by running 8 to 13 jobs at once.
