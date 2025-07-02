import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from config import Config
from tokenizer import tokenize, decode
from train_mini_gpt import MiniGPT

if __name__ == '__main__':

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        warnings.warn("CUDA is not available, using CPU")
        # Only here to allow local testing, is practically very difficult to use cpu


    config = Config()
    model = MiniGPT(config)
    model.eval()
    model.to('cuda')

    tokens = tokenize("Hello my name is") # (L)
    tokens = tokens.unsqueeze(0) # (1, L) where 1 serves as batch

    x = tokens.to('cuda')

    torch.manual_seed(1729)
    torch.cuda.manual_seed(1729)

    max_token_length = 30
    while x.size(1) < max_token_length:
        with torch.no_grad():
            logits = model(x) # (B, L, vocab_size)
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
