import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from config import Config
from primitives import SelfAttention, MLP
from dataloader import DataLoader

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
                wpe = nn.Embedding(config.block_size, config.n_embed),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embed)
            )
        )

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    def forward(self, idx, targets):
        B, L = idx.size()
        assert L <= self.config.block_size, f"Sequence length {L} must be less than block size {self.config.block_size}"

        pos = torch.arange(0, L, dtype=torch.long, device=idx.device) # (L)
        pos_emb = self.transformer.wpe(pos) # (L, n_embed)
        tok_emb = self.transformer.wte(idx) # (B, L, n_embed)

        x = tok_emb + pos_emb # (B, L, n_embed)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


if __name__ == '__main__':

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        warnings.warn("CUDA is not available, using CPU")
        # Only here to allow local testing, is practically very difficult to use cpu

    dataloader = DataLoader(10, 50, device=device)

    model = MiniGPT(Config())
    model.to(device)
    # logits, loss = model(x, y)
    # print(loss)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
        x, y = dataloader.next_batch()
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {i+1}: Loss = {loss.item()}")
