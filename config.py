from dataclasses import dataclass

@dataclass
class Config:
    block_size: int = 1024
    vocab_size: int = 50257 # Token size from GPT-2
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
