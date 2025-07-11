import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from config import Config
from tokenizer import tokenize, decode
from model import MiniGPT

def load_model_from_checkpoint(checkpoint_path, device='cpu'):
    """
    Load a MiniGPT model from a checkpoint file, handling DDP and torch.compile state_dict issues.

    Args:
        checkpoint_path (str): Path to the checkpoint file
        device (str): Device to load the model on ('cpu' or 'cuda')

    Returns:
        MiniGPT: Loaded model in eval mode
    """
    import os
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    config = Config(vocab_size=50304)  # Match the config used in training
    model = MiniGPT(config)

    # Load model from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different state_dict formats
    state_dict = checkpoint

    # If the checkpoint contains DDP module prefix, remove it
    if any(key.startswith('module.') for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # Remove 'module.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

    # Handle torch.compile wrapped models (remove _orig_mod prefix if present)
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[10:]  # Remove '_orig_mod.' prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Warning: Could not load with strict=True, trying strict=False")
        print(f"Error was: {e}")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.to(device)

    return model

if __name__ == '__main__':

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        warnings.warn("CUDA is not available, using CPU")
        # Only here to allow local testing, is practically very difficult to use cpu


    # Load model from checkpoint
    checkpoint_path = "checkpoints/checkpoint_10.pt"
    model = load_model_from_checkpoint(checkpoint_path, device)

    tokens = tokenize("Hello my name is") # (L)
    tokens = tokens.unsqueeze(0) # (1, L) where 1 serves as batch

    x = tokens.to(device)

    torch.manual_seed(1729)
    torch.cuda.manual_seed(1729)

    max_token_length = 30
    while x.size(1) < max_token_length:
        with torch.no_grad():
            logits, _ = model(x, None) # (B, L, vocab_size)
            logits = logits[:, -1, :] # Get 'newest' logit, (B, vocab_size)
            probs = F.softmax(logits, dim=-1) # convert to probs

            # Do topk sampling to ignore unlikely vals
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

            # Pick random token from topk
            ix = torch.multinomial(topk_probs, 1) # (B, 1)
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)

            # Append to the sequence to regenerate next token
            x = torch.cat((x, xcol), dim=1)
            print(decode(x[0]))
