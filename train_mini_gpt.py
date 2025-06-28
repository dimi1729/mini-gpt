import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from primitives import SelfAttention, MLP
from tokenizer import tokenize

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embed),
                wve = nn.Embedding(config.block_size, config.n_embed),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embed)
            )
        )

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    def forward(self, idx):
        B, L = idx.size()
        assert L <= self.config.block_size, f"Sequence length {L} must be less than block size {self.config.block_size}"

        pos = torch.arange(0, L, dtype=torch.long, device=idx.device) # (L)
        pos_emb = self.transformer.wpe(pos) # (L, n_embed)
        tok_emb = self.transformer.wte(idx) # (B, L, n_embed)

        x = tok_emb + pos_emb # (B, L, n_embed)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits


if __name__ == '__main__':
    config = Config()
    model = MiniGPT(config)
    model.eval()
    model.to('cuda')

    tokens = tokenize("Hello my name is") # (L)
    tokens.unsqueeze(0) # (1, L) where 1 serves as batch

    x = tokens.to('cuda')

    torch.manual_seed(1729)
    torch.cuda.manual_seed(1729)

    max_token_length = 20
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
