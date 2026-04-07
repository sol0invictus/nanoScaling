"""
SFT Evaluation Runner
=====================
Loads a trained checkpoint and runs the nanochat benchmark suite:

  Task          Type          Metric       Random baseline
  ──────────────────────────────────────────────────────
  MMLU          MC logit      accuracy          0.25
  ARC-Easy      MC logit      accuracy          0.25
  ARC-Challenge MC logit      accuracy          0.25
  HellaSwag     completion    accuracy          0.25
  GSM8K         generation    exact-match       0.00
  BPB           loss          bits/byte         n/a

Aggregate score:  ChatCORE = mean centered accuracy over
  {MMLU, ARC-Easy, ARC-Challenge, HellaSwag, GSM8K}
  where centered = (raw − baseline) / (1 − baseline)

Usage:
  python eval_sft.py out-sft/ckpt_sft.pt
  python eval_sft.py out-sft/ckpt_sft.pt --tasks mmlu arc_easy gsm8k bpb
  python eval_sft.py out-sft/ckpt_sft.pt --tasks all --max_samples 200
  python eval_sft.py out-sft/ckpt_sft.pt --tasks all --out results.json
"""

import argparse
import json
import os
import sys
import time

import torch
import tiktoken

from models import GPT
from utils.config import ExperimentConfig
from evals.tasks import ALL_TASKS

# ─── ChatCORE tasks (same set as nanochat chat_eval.py) ──────────────────────
CHATCORE_TASKS = ["mmlu", "arc_easy", "arc_challenge", "hellaswag", "gsm8k"]


# ─── Checkpoint loading ───────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: str):
    """Load a GPT model from a checkpoint saved by train.py or train_sft.py."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    cfg_dict = ckpt.get("config", {})
    if isinstance(cfg_dict, dict):
        valid_keys = set(ExperimentConfig.__dataclass_fields__.keys())
        filtered   = {k: v for k, v in cfg_dict.items() if k in valid_keys}
        if "parametrization" in filtered and isinstance(filtered["parametrization"], dict):
            from utils.config import ParametrizationConfig
            filtered["parametrization"] = ParametrizationConfig(**filtered["parametrization"])
        config = ExperimentConfig(**filtered)
    else:
        config = cfg_dict

    if not hasattr(config, "vocab_size") or config.vocab_size is None:
        config.vocab_size = cfg_dict.get("vocab_size", 50304)

    model = GPT(config)

    sd = ckpt["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd)

    model.eval()
    model.to(device)

    iter_num = ckpt.get("iter_num", "?")
    val_loss = ckpt.get("best_val_loss", "?")
    print(f"  iter={iter_num}  best_val_loss={val_loss}")
    print(f"  params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"  block_size={config.block_size}")
    return model, config


# ─── Result formatting ────────────────────────────────────────────────────────

def _fmt(v) -> str:
    if v is None:
        return "  n/a  "
    return f"{v*100:6.2f}%"


def print_results(results: list):
    SEP = "─" * 68
    print(f"\n{SEP}")
    print(f"  {'Task':<20}  {'Raw Acc':>8}  {'Centered':>9}  {'N':>6}  {'Baseline':>8}")
    print(SEP)

    chatcore_scores = []
    for r in results:
        if r.get("error"):
            print(f"  {r['task']:<20}  ERROR: {r['error'][:40]}")
            continue

        if r["task"] == "bpb":
            bpb = r.get("mean_bpb")
            bpb_str = f"{bpb:.4f}" if bpb is not None else " n/a"
            print(f"  {'BPB':<20}  {'BPB='+bpb_str:>17}  {'—':>6}  {'—':>8}")
            for corpus, cr in r.get("corpora", {}).items():
                print(f"    {corpus:<18}  {cr['bpb']:>8.4f} bpb")
            continue

        raw  = r.get("raw_accuracy")
        cent = r.get("centered_accuracy")
        n    = r.get("n", "?")
        base = r.get("random_baseline")

        print(f"  {r['task']:<20}  {_fmt(raw):>8}  {_fmt(cent):>9}  {n:>6}  {_fmt(base):>8}")

        if r["task"] in CHATCORE_TASKS and cent is not None:
            chatcore_scores.append(cent)

    print(SEP)
    if chatcore_scores:
        chatcore = sum(chatcore_scores) / len(chatcore_scores)
        print(f"  {'ChatCORE':<20}  {'':>8}  {_fmt(chatcore):>9}")
        print(f"  (mean centered accuracy over {len(chatcore_scores)} tasks)")
    print(SEP + "\n")

    return chatcore_scores


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate a nanoScaling checkpoint")
    parser.add_argument("checkpoint", help="Path to .pt checkpoint file")
    parser.add_argument(
        "--tasks", nargs="+",
        choices=list(ALL_TASKS.keys()) + ["all"],
        default=["all"],
        help="Tasks to run (default: all)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Cap samples per task (None = full dataset). Use 200 for a quick run.",
    )
    parser.add_argument(
        "--max_batches", type=int, default=100,
        help="Max batches for BPB evaluation (default: 100)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cuda / cpu (default: cuda if available)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Write JSON results to this file",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="torch.compile the model before evaluation",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, config = load_model(args.checkpoint, device)
    if args.compile:
        print("Compiling model…")
        model = torch.compile(model)

    enc = tiktoken.get_encoding("gpt2")

    task_names = list(ALL_TASKS.keys()) if "all" in args.tasks else args.tasks
    print(f"\nRunning tasks: {', '.join(task_names)}\n")

    all_results = []
    t_start = time.time()

    for task_name in task_names:
        task_cls = ALL_TASKS[task_name]

        if task_name == "bpb":
            task = task_cls(max_batches=args.max_batches)
        else:
            task = task_cls(max_samples=args.max_samples)

        print(f"\n{'─'*48}")
        print(f"  {task_name.upper()}")
        print(f"{'─'*48}")

        try:
            result = task.run(model, enc, device, config.block_size)
            all_results.append(result)

            if task_name != "bpb":
                print(f"  raw accuracy : {result['raw_accuracy']*100:.2f}%")
                print(f"  centered     : {result['centered_accuracy']*100:.2f}%")
                print(f"  n            : {result['n']}")

                if task_name == "gsm8k" and result.get("sample_failures"):
                    print("\n  Sample failures:")
                    for f in result["sample_failures"]:
                        print(f"    Q: {f['question']}")
                        print(f"    Gold: {f['gold']}  Pred: {f['pred']}")
                        print(f"    Response: {f['response'][:120]}")

        except Exception as e:
            import traceback
            print(f"  ERROR in {task_name}: {e}")
            traceback.print_exc()
            all_results.append({"task": task_name, "error": str(e)})

    elapsed = time.time() - t_start
    print(f"\nTotal eval time: {elapsed/60:.1f} min")

    chatcore_scores = print_results(all_results)

    output = {
        "checkpoint":  args.checkpoint,
        "tasks":       all_results,
        "chatcore":    sum(chatcore_scores) / len(chatcore_scores) if chatcore_scores else None,
        "elapsed_sec": elapsed,
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"Results written to {args.out}")

    return output


if __name__ == "__main__":
    main()
