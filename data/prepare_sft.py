"""
Prepare SFT dataset (dummy version).
Generates synthetic identity conversations and packs them into fixed-length blocks.
Produces:
- train.bin: input tokens
- train_mask.bin: loss mask (0 for user/system, 1 for assistant response)
- train_doc_ids.bin: document IDs for masked attention
"""

import os
import numpy as np
import tiktoken
from tqdm import tqdm

def prepare_sft(output_dir='data/sft_dummy', num_samples=1000, block_size=1024):
    os.makedirs(output_dir, exist_ok=True)
    
    enc = tiktoken.get_encoding("gpt2")
    eot_token = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    
    # Synthetic conversations
    # For now, just repeating a few patterns
    conversations = []
    for i in range(num_samples):
        conversations.append({
            "messages": [
                {"role": "user", "content": f"Who are you? {i}"},
                {"role": "assistant", "content": f"I am a synthetic assistant number {i}."},
            ]
        })
        conversations.append({
            "messages": [
                {"role": "user", "content": f"What is {i} + {i}?"},
                {"role": "assistant", "content": f"The answer is {i+i}."},
            ]
        })

    # Packing logic
    all_tokens = []
    all_masks = []   # 0 for ignore, 1 for loss
    all_doc_ids = [] # unique ID per conversation
    
    current_doc_id = 0
    
    for conv in tqdm(conversations):
        # Flatten conversation
        # Format: User: <msg> \n Assistant: <msg> <|endoftext|>
        # Simplified: <user_tokens> <assistant_tokens> <eot>
        # Loss mask: 0 on user, 1 on assistant
        
        user_text = conv['messages'][0]['content']
        assist_text = conv['messages'][1]['content']
        
        # Simple chat format
        # User: ...
        # Assistant: ...
        user_tokens = enc.encode(f"User: {user_text}\n\nAssistant: ")
        assist_tokens = enc.encode(f"{assist_text}")
        
        # Combined
        tokens = user_tokens + assist_tokens + [eot_token]
        
        # Mask: 0 for User+Prompt, 1 for Assistant, 0 for EOT (EOT is usually predicted, so maybe 1? Let's say 1 to learn to stop)
        # But commonly we mask the EOT itself if we don't want to overfit to it, but standard SFT usually trains on EOT.
        # Let's train on EOT.
        mask = [0] * len(user_tokens) + [1] * len(assist_tokens) + [1]
        
        # Doc IDs
        doc_ids = [current_doc_id] * len(tokens)
        current_doc_id += 1
        
        all_tokens.extend(tokens)
        all_masks.extend(mask)
        all_doc_ids.extend(doc_ids)
        
    # Convert to numpy and save
    # We pack into blocks of block_size for the binary files, 
    # but here we just save the linear stream. The loader will reshape.
    # Actually, for packed training, we usually want to truncate/pad to exact multiples if we treat file as 1D stream.
    # But usually we just save the stream.
    
    print(f"Total tokens: {len(all_tokens)}")
    
    # Truncate to multiple of block_size for cleanliness (optional but good for last batch)
    # Or just save all.
    
    train_ids = np.array(all_tokens, dtype=np.uint16)
    train_mask = np.array(all_masks, dtype=np.uint8)
    train_doc_ids = np.array(all_doc_ids, dtype=np.uint32) # Needs 32 bit potentially
    
    train_ids.tofile(os.path.join(output_dir, 'train.bin'))
    train_mask.tofile(os.path.join(output_dir, 'train_mask.bin'))
    train_doc_ids.tofile(os.path.join(output_dir, 'train_doc_ids.bin'))
    
    # Val data (subset of same for dummy)
    val_ids = train_ids[:1000]
    val_mask = train_mask[:1000]
    val_doc_ids = train_doc_ids[:1000]
    
    val_ids.tofile(os.path.join(output_dir, 'val.bin'))
    val_mask.tofile(os.path.join(output_dir, 'val_mask.bin'))
    val_doc_ids.tofile(os.path.join(output_dir, 'val_doc_ids.bin'))
    
    print(f"Saved to {output_dir}")

if __name__ == "__main__":
    prepare_sft()
