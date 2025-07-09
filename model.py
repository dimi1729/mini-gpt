import torch
import torch.nn as nn
import torch.nn.functional as F
from primitives import SelfAttention, MLP

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

        # Add weight sharing
        # gpt2 paper describes weight sharing between the input embedding and the output embedding
        # since the same weights should be used to translate to the embedding space
        self.transformer.wte.weight = self.lm_head.weight

        self._manual_init_weights()

    def _manual_init_weights(self):
        for module in self.modules():
            self._init_weights(module)

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "mini_gpt_scale"):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # Leave LayerNorm init alone

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
