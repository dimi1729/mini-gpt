import torch

from typing import Tuple

from tokenizer import tokenize

TRAINING_DATA_PATH = "data/tiny_shakespeare.txt"

class DataLoader:
    def __init__(self, B, T, device="cuda"):
        self.B = B
        self.T = T
        self.device = device

        self.current_position = 0

        with open(TRAINING_DATA_PATH, "r") as f:
            text = f.read()
        data = text
        self.tokens: torch.Tensor = tokenize(data)
        print(f"1 epoch = {len(self.tokens) // self.B // self.T} tokens")

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T).to(self.device)
        y = buf[1:].view(B, T).to(self.device)

        self.current_position += B*T
        if self.current_position >= len(self.tokens):
            self.current_position = 0

        return x, y
