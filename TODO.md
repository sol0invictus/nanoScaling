# Worklog

## Tasks

- [ ] Create validation dataset files
  - [ ] Run `python data/wikitext103/prepare.py`
  - [ ] Run `python data/penn_treebank/prepare.py`
  - [ ] Run `python data/pile_val/prepare.py`

- [ ] Test if validation is working
  - [ ] Run eval-only pass on a checkpoint with `configs/eval/wikitext103.yaml`
  - [ ] Confirm `metrics.json` reports correct val loss per dataset

- [ ] Create parquets for local datasets to test
  - [ ] Shakespeare: `python data/shakespeare/prepare_parquet.py` (done)
  - [ ] Any other local datasets that need converting

- [ ] Test the whole system end-to-end
  - [ ] Smoke test: short run on shakespeare parquet
  - [ ] Confirm parquet dataloader produces correct `(x, y)` batches
  - [ ] Confirm SpectralLogger and CSV logging work for RQ1/RQ2 experiment scripts
  - [ ] Confirm eval-only runs load checkpoints correctly

- [ ] Move files to cloud (what is needed)
  - [ ] Determine which data shards / checkpoints need to be uploaded
  - [ ] Sync training configs and experiment scripts

- [ ] Check DDP vs non-DDP training
  - [ ] Verify identical loss curves (single-GPU vs multi-GPU with DDP)
  - [ ] Check parquet DDP row-group sharding produces diverse batches per rank

- [ ] Try on a multi-GPU pod
  - [ ] `torchrun --standalone --nproc_per_node=N train.py <config>`
  - [ ] Monitor per-rank data loading, loss, and MFU
