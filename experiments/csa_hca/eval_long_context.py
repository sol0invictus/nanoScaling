"""
Long-context perplexity: standard attention (baseline) vs CSA.

Both 125M models in this study were trained at block_size=1024 on OpenWebText.
Neither was trained on sequences longer than 1024 tokens, so this measures
ZERO-SHOT LENGTH GENERALIZATION: perplexity at 1x / 2x / 4x the training
sequence length (1024 / 2048 / 4096). The hypothesis under test is that CSA's
architectural priors (sliding window for locality + compressed global memory)
extrapolate to longer context more gracefully than dense softmax attention,
whose RoPE positions beyond 1024 are unseen during training.

Two protocols:
  A. Stream  — OpenWebText val concatenated (EOT-separated, matching training
     packing) and chunked into non-overlapping L-token windows. Many windows
     => low-variance headline ppl. The L=1024 number cross-checks the model's
     reported best_val_loss.
  B. LongDoc — individual val documents that are themselves >= L+1 tokens; first
     L+1 tokens taken (EOT prepended as BOS). Gives a coherent same-document
     context, so the per-position NLL curve cleanly shows whether/where each
     model breaks past the 1024 training length.

Usage:
  python experiments/csa_hca/eval_long_context.py \
      --baseline out-125m-baseline/ckpt.pt --csa out-125m-csa/ckpt.pt
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from models import GPT                              # noqa: E402
from utils.config import ExperimentConfig, ParametrizationConfig  # noqa: E402

EOT = 50256  # GPT-2 end-of-text, also used as BOS in this codebase's packing


# ---------------------------------------------------------------------------- #
# model loading
# ---------------------------------------------------------------------------- #
def load_model(ckpt_path, block_size, device):
    """Rebuild a trained GPT at an *extended* block_size so it can run past the
    training length. CSA's RoPE table (persistent=False) is sized from block_size
    at construction; the baseline's fixed causal-mask buffer is dropped and
    rebuilt (the flash SDPA path ignores it, using is_causal=True)."""
    ck = torch.load(ckpt_path, map_location='cpu')
    cfgdict = ck['config']
    cfg = ExperimentConfig()
    for k, v in cfgdict.items():
        if isinstance(v, dict):
            continue  # nested config (e.g. parametrization) — keep the default object
        setattr(cfg, k, v)  # copies scalar fields incl. vocab_size (set dynamically in train.py)
    cfg.block_size = block_size
    cfg.dropout = 0.0

    model = GPT(cfg)
    sd = ck['model']
    pref = '_orig_mod.'
    sd = {(k[len(pref):] if k.startswith(pref) else k): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.endswith('.attn.bias')}  # rebuilt @ block_size
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad_missing = [m for m in missing if not m.endswith('.attn.bias')]
    assert not unexpected, f"unexpected state_dict keys: {unexpected[:8]}"
    assert not bad_missing, f"missing (non-bias) keys: {bad_missing[:8]}"
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    return model, cfg, n_params


# ---------------------------------------------------------------------------- #
# data
# ---------------------------------------------------------------------------- #
def _texts(parquet_path):
    import pyarrow.parquet as pq
    t = pq.read_table(parquet_path)
    col = 'text' if 'text' in t.column_names else t.column_names[0]
    return t.column(col).to_pylist()


def build_stream(parquet_path, cache_npy):
    """EOT-separated concatenation of all val docs -> 1D uint16 token stream."""
    if cache_npy and os.path.exists(cache_npy):
        return np.load(cache_npy)
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    out = []
    for txt in _texts(parquet_path):
        out.append(EOT)
        out.extend(enc.encode_ordinary(txt))
    arr = np.asarray(out, dtype=np.uint16)
    if cache_npy:
        np.save(cache_npy, arr)
    return arr


def build_long_docs(parquet_path, seq_len, max_docs, cache_npy):
    """Coherent docs >= seq_len+1 tokens -> (N, seq_len+1) int array, EOT BOS."""
    if cache_npy and os.path.exists(cache_npy):
        return np.load(cache_npy)
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    rows = []
    for txt in _texts(parquet_path):
        ids = enc.encode_ordinary(txt)
        if len(ids) >= seq_len:                    # need seq_len content toks
            rows.append([EOT] + ids[:seq_len])     # length seq_len+1
            if len(rows) >= max_docs:
                break
    arr = np.asarray(rows, dtype=np.int64)
    if cache_npy:
        np.save(cache_npy, arr)
    return arr


# ---------------------------------------------------------------------------- #
# evaluation
# ---------------------------------------------------------------------------- #
@torch.no_grad()
def _eval_windows(model, windows, L, batch, device, amp_ctx):
    """windows: (N, L+1) int array. Returns mean NLL, ppl, per-position NLL."""
    pos_loss = np.zeros(L, dtype=np.float64)
    n_win = windows.shape[0]
    total_loss, total_tok = 0.0, 0
    for b0 in range(0, n_win, batch):
        chunk = windows[b0:b0 + batch]
        x = torch.from_numpy(chunk[:, :-1]).to(device)
        y = torch.from_numpy(chunk[:, 1:]).to(device)
        with amp_ctx:
            logits, _, _ = model(x, targets=y)     # full logits; ignore bundled aux loss
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
            reduction='none').reshape(x.size(0), L)
        pos_loss += loss.sum(0).double().cpu().numpy()
        total_loss += float(loss.sum().item())
        total_tok += loss.numel()
    pos_nll = pos_loss / n_win
    mean_nll = total_loss / total_tok
    return dict(mean_nll=mean_nll, ppl=float(np.exp(mean_nll)),
                pos_nll=pos_nll.tolist(), n_windows=int(n_win), n_tokens=int(total_tok))


def stream_windows(stream, L, max_windows):
    n = (len(stream) - 1) // L
    starts = np.arange(n) * L
    if max_windows and n > max_windows:
        starts = starts[np.linspace(0, n - 1, max_windows).astype(int)]
    return np.stack([stream[s:s + L + 1] for s in starts]).astype(np.int64)


def batch_for(L, base):
    """Scale batch down with length to stay within memory (logits are O(B*L*V))."""
    return max(1, base // max(1, L // 1024))


# ---------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', default='out-125m-baseline/ckpt.pt')
    ap.add_argument('--csa', default='out-125m-csa/ckpt.pt')
    ap.add_argument('--parquet', default='training_data/openwebtext/shard_00321.parquet',
                    help='OpenWebText val shard (last shard = val by convention)')
    ap.add_argument('--lengths', type=int, nargs='+', default=[1024, 2048, 4096])
    ap.add_argument('--max_windows', type=int, default=512,
                    help='cap stream windows per length (evenly spaced) for speed')
    ap.add_argument('--max_docs', type=int, default=128, help='coherent docs per length (protocol B)')
    ap.add_argument('--batch', type=int, default=16, help='batch at L=1024; halved per 2x length')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    ap.add_argument('--out', default='experiments/csa_hca/long_context_results.json')
    ap.add_argument('--plot', default='experiments/csa_hca/long_context_ppl.png')
    args = ap.parse_args()

    os.chdir(ROOT)
    device = args.device
    maxL = max(args.lengths)
    amp_ctx = (torch.autocast(device_type='cuda', dtype=getattr(torch, args.dtype))
               if device == 'cuda' and args.dtype != 'float32' else torch.no_grad())

    print(f"device={device} dtype={args.dtype} maxL={maxL} lengths={args.lengths}")
    print("building eval data from", args.parquet, "...")
    t0 = time.time()
    stream = build_stream(args.parquet, 'experiments/csa_hca/_owtval_stream.npy')
    print(f"  stream: {len(stream):,} tokens  ({time.time()-t0:.1f}s)")
    long_docs = {L: build_long_docs(args.parquet, L, args.max_docs,
                                    f'experiments/csa_hca/_owtval_docs_{L}.npy')
                 for L in args.lengths}
    for L in args.lengths:
        print(f"  long docs >= {L}t : {long_docs[L].shape[0]} docs")

    results = {'config': vars(args), 'models': {}}
    for name, ckpt in [('baseline', args.baseline), ('csa', args.csa)]:
        print(f"\n=== {name}  ({ckpt}) ===")
        model, cfg, n_params = load_model(ckpt, maxL, device)
        print(f"  loaded: {n_params/1e6:.1f}M params, use_csa={cfg.use_csa}, block_size->{cfg.block_size}")
        m = {'n_params': n_params, 'use_csa': bool(cfg.use_csa), 'stream': {}, 'longdoc': {}}
        for L in args.lengths:
            bs = batch_for(L, args.batch)
            # Protocol A: stream
            w = stream_windows(stream, L, args.max_windows)
            t0 = time.time()
            r = _eval_windows(model, w, L, bs, device, amp_ctx)
            m['stream'][L] = r
            print(f"  [stream ] L={L:>4}  ppl={r['ppl']:7.2f}  nll={r['mean_nll']:.4f}"
                  f"  ({r['n_windows']} win, bs={bs}, {time.time()-t0:.1f}s)")
            # Protocol B: coherent long docs
            docs = long_docs[L]
            if docs.shape[0] >= 8:
                t0 = time.time()
                rd = _eval_windows(model, docs, L, bs, device, amp_ctx)
                m['longdoc'][L] = rd
                print(f"  [longdoc] L={L:>4}  ppl={rd['ppl']:7.2f}  nll={rd['mean_nll']:.4f}"
                      f"  ({rd['n_windows']} docs, {time.time()-t0:.1f}s)")
        results['models'][name] = m
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")

    _summary(results, args.lengths)
    _plots(results, args.lengths, args.plot)


def _summary(results, lengths):
    b, c = results['models']['baseline'], results['models']['csa']
    print("\n" + "=" * 64)
    print("PERPLEXITY  (OpenWebText val, zero-shot length generalization)")
    print("=" * 64)
    for proto, label in [('stream', 'Protocol A: concatenated stream'),
                         ('longdoc', 'Protocol B: coherent long docs')]:
        print(f"\n{label}")
        print(f"  {'context':>8} | {'baseline':>10} | {'CSA':>10} | {'Δ (CSA-base)':>13} | winner")
        print("  " + "-" * 60)
        for L in lengths:
            if L not in b[proto] or L not in c[proto]:
                continue
            pb, pc = b[proto][L]['ppl'], c[proto][L]['ppl']
            mult = L // lengths[0]
            win = 'baseline' if pb < pc else 'CSA'
            print(f"  {L:>5}({mult}x) | {pb:>10.2f} | {pc:>10.2f} | {pc-pb:>+13.2f} | {win}")
    # degradation ratio vs 1x
    print("\nPpl inflation relative to 1x (stream): how much worse at longer context")
    print(f"  {'context':>8} | {'baseline':>10} | {'CSA':>10}")
    print("  " + "-" * 34)
    L0 = lengths[0]
    for L in lengths:
        if L not in b['stream']:
            continue
        rb = b['stream'][L]['ppl'] / b['stream'][L0]['ppl']
        rc = c['stream'][L]['ppl'] / c['stream'][L0]['ppl']
        print(f"  {L:>8} | {rb:>9.2f}x | {rc:>9.2f}x")


def _plots(results, lengths, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib unavailable, skipping plots:", e)
        return
    b, c = results['models']['baseline'], results['models']['csa']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (1) ppl vs context length, both protocols
    ax = axes[0]
    for proto, ls in [('stream', '-o'), ('longdoc', '--s')]:
        Ls = [L for L in lengths if L in b[proto] and L in c[proto]]
        ax.plot(Ls, [b[proto][L]['ppl'] for L in Ls], ls, color='tab:blue',
                label=f'baseline ({proto})')
        ax.plot(Ls, [c[proto][L]['ppl'] for L in Ls], ls, color='tab:red',
                label=f'CSA ({proto})')
    ax.axvline(1024, color='gray', ls=':', lw=1)
    ax.text(1024, ax.get_ylim()[1], ' train len', color='gray', va='top', fontsize=8)
    ax.set_xlabel('context length (tokens)'); ax.set_ylabel('perplexity')
    ax.set_xticks(lengths); ax.set_title('Perplexity vs context length'); ax.legend(fontsize=8)

    # (2) per-position NLL at max length (coherent docs = cleanest signal)
    ax = axes[1]
    Lmax = max(L for L in lengths)
    src = 'longdoc' if Lmax in b.get('longdoc', {}) else 'stream'
    if Lmax in b[src] and Lmax in c[src]:
        def smooth(v, k=33):
            v = np.asarray(v); ker = np.ones(k) / k
            return np.convolve(v, ker, mode='valid')
        pb, pc = b[src][Lmax]['pos_nll'], c[src][Lmax]['pos_nll']
        xs = np.arange(len(pb))[16:-16]
        ax.plot(xs, smooth(pb), color='tab:blue', label='baseline')
        ax.plot(xs, smooth(pc), color='tab:red', label='CSA')
        ax.axvline(1024, color='gray', ls=':', lw=1)
        ax.text(1024, ax.get_ylim()[1], ' train len', color='gray', va='top', fontsize=8)
        ax.set_xlabel('token position in sequence'); ax.set_ylabel('NLL (nats, smoothed)')
        ax.set_title(f'Per-position loss @ L={Lmax}  ({src})'); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


if __name__ == '__main__':
    main()
