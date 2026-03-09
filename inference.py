"""
Model inference module for nanoScaling.

Load a trained checkpoint and generate text with KV-cache support.

Usage (CLI):
    # From a nanoScaling checkpoint
    python inference.py --checkpoint out/ckpt.pt --prompt "Once upon a time"

    # From a GPT-2 pretrained model
    python inference.py --init_from gpt2 --prompt "The meaning of life is"

    # With sampling parameters
    python inference.py --checkpoint out/ckpt.pt \
        --prompt "Hello world" \
        --max_new_tokens 200 \
        --temperature 0.8 \
        --top_k 50 \
        --top_p 0.95

    # Disable KV caching (slower but useful for debugging)
    python inference.py --checkpoint out/ckpt.pt --prompt "Test" --no_kv_cache

Usage (Python API):
    from inference import ModelInference

    engine = ModelInference.from_checkpoint("out/ckpt.pt")
    text = engine.generate("Once upon a time", max_new_tokens=100)
    print(text)
"""

import os
import time
import pickle
import argparse
from contextlib import nullcontext

import torch
import tiktoken

from models.gpt import GPT, GPTConfig


class ModelInference:
    """High-level inference wrapper around a GPT model with KV-cache support."""

    def __init__(self, model, encode_fn, decode_fn, device='cpu', dtype='bfloat16'):
        """
        Args:
            model: a GPT model instance (already on device, in eval mode)
            encode_fn: callable that maps a string to a list of token ids
            decode_fn: callable that maps a list of token ids to a string
            device: torch device string
            dtype: 'float32', 'bfloat16', or 'float16'
        """
        self.model = model
        self.encode = encode_fn
        self.decode = decode_fn
        self.device = device
        self.dtype = dtype

        dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
        self._ptdtype = dtype_map.get(dtype, torch.bfloat16)
        device_type = 'cuda' if 'cuda' in device else 'cpu'
        self._ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=self._ptdtype)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None, dtype=None, compile_model=False):
        """
        Load a nanoScaling checkpoint and return a ready-to-use ModelInference.

        Args:
            checkpoint_path: path to a ckpt.pt file
            device: device string (auto-detected if None)
            dtype: dtype string (auto-detected if None)
            compile_model: whether to torch.compile the model
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if dtype is None:
            if device == 'cpu':
                dtype = 'float32'
            elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = 'bfloat16'
            else:
                dtype = 'float16'

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Build model config
        model_args = checkpoint['model_args']
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

        # Strip torch.compile prefix if present
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)

        if compile_model:
            model = torch.compile(model)

        # Set up tokenizer
        encode_fn, decode_fn = _build_tokenizer(checkpoint)

        return cls(model, encode_fn, decode_fn, device=device, dtype=dtype)

    @classmethod
    def from_pretrained(cls, model_type='gpt2', device=None, dtype=None, compile_model=False):
        """
        Load a HuggingFace GPT-2 pretrained model.

        Args:
            model_type: one of 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
            device: device string
            dtype: dtype string
            compile_model: whether to torch.compile the model
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if dtype is None:
            if device == 'cpu':
                dtype = 'float32'
            elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = 'bfloat16'
            else:
                dtype = 'float16'

        model = GPT.from_pretrained(model_type, dict(dropout=0.0))
        model.eval()
        model.to(device)

        if compile_model:
            model = torch.compile(model)

        enc = tiktoken.get_encoding("gpt2")
        encode_fn = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode_fn = lambda l: enc.decode(l)

        return cls(model, encode_fn, decode_fn, device=device, dtype=dtype)

    def generate(self, prompt, max_new_tokens=100, temperature=0.8, top_k=200,
                 top_p=None, use_kv_cache=True, seed=None):
        """
        Generate text from a prompt string.

        Args:
            prompt: input text string
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering (None to disable)
            top_p: nucleus sampling threshold (None to disable)
            use_kv_cache: use KV caching for faster generation
            seed: random seed for reproducibility (None for non-deterministic)

        Returns:
            generated text string (prompt + completion)
        """
        if seed is not None:
            torch.manual_seed(seed)
            if 'cuda' in self.device:
                torch.cuda.manual_seed(seed)

        start_ids = self.encode(prompt)
        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]

        with torch.no_grad():
            with self._ctx:
                y = self.model.generate(
                    x, max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    use_kv_cache=use_kv_cache,
                )

        return self.decode(y[0].tolist())

    def generate_batch(self, prompts, max_new_tokens=100, temperature=0.8,
                       top_k=200, top_p=None, use_kv_cache=True):
        """
        Generate text for multiple prompts. Each prompt is processed independently
        (no padding/batching across different-length prompts).

        Args:
            prompts: list of input text strings
            max_new_tokens: number of tokens to generate per prompt
            temperature: sampling temperature
            top_k: top-k filtering
            top_p: nucleus sampling threshold
            use_kv_cache: use KV caching

        Returns:
            list of generated text strings
        """
        return [
            self.generate(p, max_new_tokens=max_new_tokens, temperature=temperature,
                          top_k=top_k, top_p=top_p, use_kv_cache=use_kv_cache)
            for p in prompts
        ]

    def benchmark(self, prompt="The", max_new_tokens=100, use_kv_cache=True, warmup_tokens=20):
        """
        Benchmark generation speed (tokens/sec) with and without KV cache.

        Returns:
            dict with 'tokens_per_sec', 'total_time_ms', 'max_new_tokens', 'use_kv_cache'
        """
        # Warmup
        _ = self.generate(prompt, max_new_tokens=warmup_tokens, use_kv_cache=use_kv_cache)

        if 'cuda' in self.device:
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        _ = self.generate(prompt, max_new_tokens=max_new_tokens, use_kv_cache=use_kv_cache)
        if 'cuda' in self.device:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000
        tokens_per_sec = max_new_tokens / (t1 - t0)

        return {
            'tokens_per_sec': tokens_per_sec,
            'total_time_ms': elapsed_ms,
            'max_new_tokens': max_new_tokens,
            'use_kv_cache': use_kv_cache,
        }


def _build_tokenizer(checkpoint):
    """Build encode/decode functions from a checkpoint, falling back to GPT-2 tokenizer."""
    # Check for a dataset-specific meta.pkl
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            encode_fn = lambda s: [stoi[c] for c in s]
            decode_fn = lambda l: ''.join([itos[i] for i in l])
            return encode_fn, decode_fn

    # Default: GPT-2 BPE tokenizer
    enc = tiktoken.get_encoding("gpt2")
    encode_fn = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode_fn = lambda l: enc.decode(l)
    return encode_fn, decode_fn


def main():
    parser = argparse.ArgumentParser(description='Generate text from a nanoScaling model checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file (ckpt.pt)')
    parser.add_argument('--init_from', type=str, default=None,
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='Load a pretrained GPT-2 model instead of a checkpoint')
    parser.add_argument('--prompt', type=str, default='\n',
                        help='Text prompt to condition generation on')
    parser.add_argument('--prompt_file', type=str, default=None,
                        help='Read prompt from a file instead of --prompt')
    parser.add_argument('--max_new_tokens', type=int, default=200,
                        help='Number of tokens to generate')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of independent samples to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=200,
                        help='Top-k sampling (0 to disable)')
    parser.add_argument('--top_p', type=float, default=None,
                        help='Nucleus (top-p) sampling threshold')
    parser.add_argument('--no_kv_cache', action='store_true',
                        help='Disable KV caching (slower)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (auto-detected if not set)')
    parser.add_argument('--dtype', type=str, default=None,
                        choices=['float32', 'bfloat16', 'float16'],
                        help='Data type (auto-detected if not set)')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile (requires PyTorch 2.0+)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run a generation speed benchmark')
    args = parser.parse_args()

    # Load model
    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        engine = ModelInference.from_checkpoint(
            args.checkpoint, device=args.device, dtype=args.dtype, compile_model=args.compile
        )
    elif args.init_from is not None:
        print(f"Loading pretrained: {args.init_from}")
        engine = ModelInference.from_pretrained(
            args.init_from, device=args.device, dtype=args.dtype, compile_model=args.compile
        )
    else:
        parser.error("Must specify either --checkpoint or --init_from")

    # Read prompt
    prompt = args.prompt
    if args.prompt_file is not None:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read()

    top_k = args.top_k if args.top_k > 0 else None
    use_kv_cache = not args.no_kv_cache

    if args.benchmark:
        print("\nBenchmarking with KV cache...")
        result = engine.benchmark(prompt, max_new_tokens=args.max_new_tokens, use_kv_cache=True)
        print(f"  KV cache ON:  {result['tokens_per_sec']:.1f} tokens/sec  ({result['total_time_ms']:.0f} ms for {result['max_new_tokens']} tokens)")

        print("Benchmarking without KV cache...")
        result = engine.benchmark(prompt, max_new_tokens=args.max_new_tokens, use_kv_cache=False)
        print(f"  KV cache OFF: {result['tokens_per_sec']:.1f} tokens/sec  ({result['total_time_ms']:.0f} ms for {result['max_new_tokens']} tokens)")
        return

    # Generate
    for i in range(args.num_samples):
        text = engine.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
            top_p=args.top_p,
            use_kv_cache=use_kv_cache,
            seed=args.seed,
        )
        print(text)
        if args.num_samples > 1:
            print('---------------')


if __name__ == '__main__':
    main()
