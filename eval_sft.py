"""
SFT Evaluation Script
Loads a trained model and generates responses.
"""
import os
import sys
import torch
import tiktoken
from models import GPT, GPTConfig
from utils.config import ExperimentConfig

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
out_dir = 'out-sft'
device = 'cuda'
num_samples = 5
max_new_tokens = 50
temperature = 0.8
top_k = 200
seed = 1337

# Override settings
if len(sys.argv) > 1:
    out_dir = sys.argv[1]

ckpt_path = os.path.join(out_dir, 'ckpt_sft.pt')
if not os.path.exists(ckpt_path):
    ckpt_path = os.path.join(out_dir, 'ckpt.pt') # fallback

print(f"Loading checkpoint from {ckpt_path}")

checkpoint = torch.load(ckpt_path, map_location=device)
config_dict = checkpoint['config']
iter_num = checkpoint['iter_num']

# Fix legacy config keys if any
if 'parametrization' in config_dict and isinstance(config_dict['parametrization'], dict) and 'mode' in config_dict['parametrization']:
     # Reconstruct the ParametrizationConfig object or just let the model handle simple settings
     pass

# Create config object (relying on correct dict structure)
# We might need to handle nested dataclasses carefully if we use ExperimentConfig directly
# But GPT model expects GPTConfig usually? 
# No, in this codebase GPT takes ExperimentConfig or GPTConfig?
# train.py uses ExperimentConfig but GPT __init__ takes "config".
# GPTConfig is a dataclass in models/gpt.py.
# ExperimentConfig in utils/config.py contains the fields.
# Let's try to reconstruct ExperimentConfig.
# Filter out keys that are not in ExperimentConfig
valid_keys = ExperimentConfig.__dict__['__dataclass_fields__'].keys()
filtered_config_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
config = ExperimentConfig(**filtered_config_dict)
# Restore extra keys manually
if 'vocab_size' in config_dict:
    config.vocab_size = config_dict['vocab_size']
else:
    config.vocab_size = 50304

model = GPT(config)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if config.compile:
    model = torch.compile(model)

enc = tiktoken.get_encoding("gpt2")

# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------

prompts = [
    "User: Who are you?\n\nAssistant:",
    "User: What is 1 + 1?\n\nAssistant:",
    "User: Tell me a story.\n\nAssistant:",
    "User: What is 5 + 5?\n\nAssistant:",
]

print(f"Generating {num_samples} samples per prompt with temp {temperature}...")

with torch.no_grad():
    for prompt in prompts:
        print("\n" + "-"*80)
        print(f"Prompt: {prompt.strip()}")
        print("-" * 80)
        
        start_ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        
        # Simple generation loop
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        
        output_text = enc.decode(y[0].tolist())
        print(output_text)
        print("-" * 80)

