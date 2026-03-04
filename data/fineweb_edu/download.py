"""
Download FineWeb-Edu 100B pre-shuffled parquet shards from HuggingFace.

Source dataset: karpathy/fineweb-edu-100b-shuffle (1823 shards)
Each shard is ~100MB of raw text in parquet format with a 'text' column.

These shards are ready to use directly with the nanoScaling parquet dataloader
(utils/dataloader.py) — no further preparation needed.

Usage:
    # Download first 10 shards (enough for a quick experiment)
    python data/fineweb_edu/download.py -n 10

    # Download 50 shards with 8 parallel workers to a custom directory
    python data/fineweb_edu/download.py -n 50 -w 8 -d /data/fineweb_edu

    # Download shards 100-149 (resume / fill gaps)
    python data/fineweb_edu/download.py -n 50 --start_index 100

    # Download all 1823 shards (~180GB)
    python data/fineweb_edu/download.py -n 1823 -w 8
"""

import argparse
import os
import time
from multiprocessing import Pool

import requests

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822  # shards are 0000 .. 1822 inclusive
DEFAULT_OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _shard_filename(index: int) -> str:
    return f"shard_{index:05d}.parquet"


def _shard_url(index: int) -> str:
    return f"{BASE_URL}/{_shard_filename(index)}"


# ---------------------------------------------------------------------------
# Download worker
# ---------------------------------------------------------------------------

def download_single_file(args) -> bool:
    """Download one shard with retry logic and atomic write.

    Args:
        args: (index, output_dir) tuple — must be a tuple for multiprocessing.Pool.

    Returns:
        True on success, False on permanent failure.
    """
    index, output_dir = args
    filename = _shard_filename(index)
    out_path = os.path.join(output_dir, filename)
    tmp_path = out_path + ".tmp"

    if os.path.exists(out_path):
        print(f"  Skipping {filename} (already exists)")
        return True

    url = _shard_url(index)
    max_attempts = 5

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                    if chunk:
                        f.write(chunk)
            os.rename(tmp_path, out_path)  # atomic write
            print(f"  Downloaded {filename}")
            return True

        except (requests.RequestException, IOError, OSError) as e:
            # Clean up any partial file
            for path in [tmp_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

            if attempt < max_attempts:
                wait = 2 ** attempt
                print(f"  Retry {attempt}/{max_attempts} for {filename}: {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                print(f"  FAILED to download {filename} after {max_attempts} attempts: {e}")
                return False

    return False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def list_parquet_files(data_dir: str = None) -> list:
    """Return sorted list of shard_?????.parquet paths in data_dir."""
    import glob
    data_dir = data_dir or DEFAULT_OUTPUT_DIR
    pattern = os.path.join(data_dir, "shard_?????.parquet")
    return sorted(glob.glob(pattern))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download fineweb-edu-100b-shuffle parquet shards from HuggingFace.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", "--num_files",
        type=int, default=10,
        help="Number of shards to download",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int, default=4,
        help="Parallel download workers",
    )
    parser.add_argument(
        "-d", "--output_dir",
        type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory to save parquet files",
    )
    parser.add_argument(
        "--start_index",
        type=int, default=0,
        help="Index of the first shard to download (0-based)",
    )
    args = parser.parse_args()

    # Validate
    if args.start_index < 0 or args.start_index > MAX_SHARD:
        parser.error(f"--start_index must be between 0 and {MAX_SHARD}")
    if args.num_files < 1:
        parser.error("--num_files must be >= 1")

    end_index = min(args.start_index + args.num_files - 1, MAX_SHARD)
    indices = list(range(args.start_index, end_index + 1))

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Downloading shards {indices[0]:05d}–{indices[-1]:05d} "
          f"({len(indices)} files) to {args.output_dir}")
    print(f"Using {args.workers} parallel worker(s).")
    print()

    tasks = [(i, args.output_dir) for i in indices]

    if args.workers > 1:
        with Pool(processes=args.workers) as pool:
            results = pool.map(download_single_file, tasks)
    else:
        results = [download_single_file(t) for t in tasks]

    successful = sum(1 for r in results if r)
    failed = len(results) - successful

    print()
    print(f"Done. {successful}/{len(results)} shards downloaded to {args.output_dir}.")
    if failed:
        print(f"WARNING: {failed} shard(s) failed. Re-run to retry.")

    existing = list_parquet_files(args.output_dir)
    if existing:
        print(f"\nDataset summary ({len(existing)} shards in {args.output_dir}):")
        print(f"  Train split: {len(existing) - 1} shard(s) (shard_00000 … shard_{len(existing)-2:05d})")
        print(f"  Val split:   1 shard  (shard_{len(existing)-1:05d})")
        print(f"\nTo start training: set dataset=fineweb_edu in your config yaml.")


if __name__ == "__main__":
    main()
