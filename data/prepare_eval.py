"""
Prepare all data needed for eval_sft.py.

Downloads and caches:
  - MMLU          (cais/mmlu)                   — 57 subjects, HF cache
  - ARC-Easy      (allenai/ai2_arc, ARC-Easy)   — HF cache
  - ARC-Challenge (allenai/ai2_arc, ARC-Challenge) — HF cache
  - HellaSwag     (Rowan/hellaswag)              — HF cache
  - GSM8K         (openai/gsm8k)                 — HF cache
  - WikiText-103  (wikitext-103-raw-v1)          — local parquet in data/wikitext103/
  - Pile-val      (monology/pile-uncopyrighted)  — local parquet in data/pile_val/

Usage:
  python data/prepare_eval.py               # full download
  python data/prepare_eval.py --skip_bpb    # skip BPB corpora (fast, HF tasks only)
  python data/prepare_eval.py --tasks mmlu arc hellaswag gsm8k
"""

import argparse
import os
import subprocess
import sys

# ─── HuggingFace dataset pre-download ─────────────────────────────────────────

def download_mmlu():
    print("\n[MMLU] Downloading all 57 subjects from cais/mmlu...")
    from datasets import load_dataset

    subjects = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology",
        "high_school_statistics", "high_school_us_history",
        "high_school_world_history", "human_aging", "human_sexuality",
        "international_law", "jurisprudence", "logical_fallacies",
        "machine_learning", "management", "marketing", "medical_genetics",
        "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
        "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology",
        "us_foreign_policy", "virology", "world_religions",
    ]

    total_test = 0
    failed = []
    for i, subject in enumerate(subjects):
        try:
            dev  = load_dataset("cais/mmlu", subject, split="dev",  trust_remote_code=True)
            test = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
            total_test += len(test)
            print(f"  [{i+1:2d}/57] {subject}: {len(test)} test, {len(dev)} dev")
        except Exception as e:
            print(f"  [{i+1:2d}/57] {subject}: FAILED — {e}")
            failed.append(subject)

    print(f"\n  MMLU: {total_test:,} test examples across {len(subjects)-len(failed)} subjects")
    if failed:
        print(f"  Failed subjects: {failed}")
    return len(failed) == 0


def download_arc():
    print("\n[ARC] Downloading allenai/ai2_arc (Easy + Challenge)...")
    from datasets import load_dataset

    ok = True
    for config_name in ["ARC-Easy", "ARC-Challenge"]:
        try:
            ds = load_dataset("allenai/ai2_arc", config_name, split="test", trust_remote_code=True)
            print(f"  {config_name}: {len(ds):,} test examples")
        except Exception as e:
            print(f"  {config_name}: FAILED — {e}")
            ok = False
    return ok


def download_hellaswag():
    print("\n[HellaSwag] Downloading Rowan/hellaswag...")
    from datasets import load_dataset
    try:
        ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
        print(f"  HellaSwag validation: {len(ds):,} examples")
        return True
    except Exception as e:
        print(f"  FAILED — {e}")
        return False


def download_gsm8k():
    print("\n[GSM8K] Downloading openai/gsm8k...")
    from datasets import load_dataset
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
        print(f"  GSM8K test: {len(ds):,} examples")
        return True
    except Exception as e:
        print(f"  FAILED — {e}")
        return False


# ─── BPB local corpus preparation ─────────────────────────────────────────────

def _run_prepare(script_path: str, extra_args: list = []) -> bool:
    """Run a data/*/prepare.py script as a subprocess."""
    cmd = [sys.executable, script_path] + extra_args
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def prepare_wikitext103():
    print("\n[BPB / WikiText-103] Preparing local parquet...")
    script = os.path.join(os.path.dirname(__file__), "wikitext103", "prepare.py")
    out_val = os.path.join(os.path.dirname(__file__), "wikitext103", "shard_00001.parquet")
    if os.path.isfile(out_val):
        size_mb = os.path.getsize(out_val) / 1e6
        print(f"  Already exists ({size_mb:.1f} MB) — skipping.")
        return True
    return _run_prepare(script)


def prepare_pile_val():
    print("\n[BPB / Pile-val] Preparing local parquet (streams first 10k docs)...")
    script = os.path.join(os.path.dirname(__file__), "pile_val", "prepare.py")
    out = os.path.join(os.path.dirname(__file__), "pile_val", "shard_00000.parquet")
    if os.path.isfile(out):
        size_mb = os.path.getsize(out) / 1e6
        print(f"  Already exists ({size_mb:.1f} MB) — skipping.")
        return True
    return _run_prepare(script)


# ─── Main ─────────────────────────────────────────────────────────────────────

ALL_TASKS = ["mmlu", "arc", "hellaswag", "gsm8k", "bpb"]

TASK_FNS = {
    "mmlu":      download_mmlu,
    "arc":       download_arc,
    "hellaswag": download_hellaswag,
    "gsm8k":     download_gsm8k,
}

BPB_FNS = [
    ("wikitext103", prepare_wikitext103),
    ("pile_val",    prepare_pile_val),
]


def main():
    parser = argparse.ArgumentParser(description="Pre-download all eval data for eval_sft.py")
    parser.add_argument(
        "--tasks", nargs="+",
        choices=ALL_TASKS + ["all"],
        default=["all"],
        help="Which datasets to download (default: all)",
    )
    parser.add_argument(
        "--skip_bpb", action="store_true",
        help="Skip BPB local corpus preparation (WikiText-103, Pile-val)",
    )
    args = parser.parse_args()

    tasks = set(ALL_TASKS if "all" in args.tasks else args.tasks)
    if args.skip_bpb:
        tasks.discard("bpb")

    print("=" * 60)
    print("  eval_sft.py data preparation")
    print("=" * 60)

    results = {}

    # HuggingFace tasks
    for name, fn in TASK_FNS.items():
        if name in tasks:
            results[name] = fn()

    # BPB local corpora
    if "bpb" in tasks:
        print("\n[BPB] Preparing local evaluation corpora...")
        bpb_ok = True
        for name, fn in BPB_FNS:
            ok = fn()
            results[f"bpb/{name}"] = ok
            if not ok:
                bpb_ok = False
        results["bpb"] = bpb_ok

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    all_ok = True
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name:<25} {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll data ready. Run eval with:")
        print("  python eval_sft.py <checkpoint.pt>")
    else:
        print("\nSome downloads failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
