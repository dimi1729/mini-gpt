import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import time

from config import Config
from primitives import SelfAttention, MLP
from dataloader import DataLoader
from learning_rate_scheduler import LearningRateScheduler

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


if __name__ == '__main__':

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        warnings.warn("CUDA is not available, using CPU")
        # Only here to allow local testing, is practically very difficult to use cpu

    torch.manual_seed(1729)
    torch.cuda.manual_seed(1729)

    # Use gpt3 ~0.5M batch size with grad accumulation
    total_batch_size = 524288  # 2**19
    B = 4
    T = 1024
    assert total_batch_size % (B * T) == 0, "Batch size must be divisible by B"
    grad_accum_steps = total_batch_size // (B * T)

    dataloader = DataLoader(B=B, T=T, device=device)

    torch.set_float32_matmul_precision('high') # TF32 precision

    config = Config(vocab_size=50304) # 50304 is div by 128 so it is a "nicer number" than 50257
    model = MiniGPT(config)
    model.to(device)
    model = torch.compile(model)
    # logits, loss = model(x, y)
    # print(loss)

    initial_optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, fused=True)
    lr_scheduler = LearningRateScheduler(initial_optimizer)
    for i in range(50):
        t0 = time.time()
        lr_scheduler.optimizer.zero_grad()

        total_loss = 0.0
        for sub_step in range(grad_accum_steps):
            x, y = dataloader.next_batch()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            total_loss += loss.detach()
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr_scheduler.set_lr(i)
        lr_scheduler.optimizer.step()

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tokens_per_s = (dataloader.B * dataloader.T) / dt
        print(f"Epoch {i+1}: Loss = {total_loss}, time = {dt*1000:.2f} ms, tok/s = {tokens_per_s:.2f}, norm = {norm:.3f}, lr = {lr_scheduler.get_lr(i):.5f}")
