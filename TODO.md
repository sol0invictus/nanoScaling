# Worklog

## Tasks

- [x] Create validation dataset files
  - [x] Run `python data/wikitext103/prepare.py`
  - [x] Run `python data/penn_treebank/prepare.py`
  - [x] Run `python data/pile_val/prepare.py`

- [x] Test if validation is working
  - [x] Run eval-only pass on a checkpoint with `configs/eval/wikitext103.yaml`
  - [x] Confirm `metrics.json` reports correct val loss per dataset

- [x] Create parquets for local datasets to test
  - [x] Shakespeare: `python data/shakespeare/prepare_parquet.py` (done)
  - [x] Any other local datasets that need converting

- [x] Test the whole system end-to-end
  - [x] Smoke test: short run on shakespeare parquet
  - [x] Confirm parquet dataloader produces correct `(x, y)` batches
  - [x] Confirm SpectralLogger and CSV logging work for RQ1/RQ2 experiment scripts
  - [x] Confirm eval-only runs load checkpoints correctly

- [ ] Move files to cloud (what is needed)
  - [ ] Determine which data shards / checkpoints need to be uploaded
  - [ ] Sync training configs and experiment scripts

- [x] Check DDP vs non-DDP training
  - [x] Verify identical loss curves (single-GPU vs multi-GPU with DDP)
  - [x] Check parquet DDP row-group sharding produces diverse batches per rank

- [ ] Try on a multi-GPU pod
  - [ ] `torchrun --standalone --nproc_per_node=N train.py <config>`
  - [ ] Monitor per-rank data loading, loss, and MFU
