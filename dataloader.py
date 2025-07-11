import torch

from typing import Tuple

from tokenizer import tokenize

TRAINING_DATA_PATH = "data/tiny_shakespeare.txt"  # 338025 total tokens

class DataLoader:
    def __init__(self, B, T, process_rank, num_processes, device="cuda"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.device = device

        self.current_position = B * T * process_rank

        with open(TRAINING_DATA_PATH, "r") as f:
            text = f.read()
        data = text
        self.tokens: torch.Tensor = tokenize(data)

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        if len(buf) < B * T + 1:
            self.current_position = 0
            buf = self.tokens[self.current_position:self.current_position + B * T + 1]

        x = buf[:-1].view(B, T).to(self.device)
        y = buf[1:].view(B, T).to(self.device)

        self.current_position += B * T * self.num_processes
        if self.current_position >= len(self.tokens):
            self.current_position = B * T * self.process_rank

        return x, y

if __name__ == "__main__":
    # Quick way to see num tokens in dataset
    dataloader = DataLoader(4, 8, 0, 1, "cuda")
    print(dataloader.tokens.shape)
