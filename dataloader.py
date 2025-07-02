import torch

from typing import Tuple

from tokenizer import decode, tokenize

TRAINING_DATA_PATH = "data/tiny_shakespeare.txt"
NUM_BATCHES = 2
NUM_TOKENS = 5

def create_training_batches(training_data_path: str = TRAINING_DATA_PATH,
                            B: int = NUM_BATCHES,
                            T: int = NUM_TOKENS,
                            device = "cuda"
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
    with open(training_data_path, "r") as f:
        text = f.read()
    data = text

    tokens = tokenize(data)

    if B > 1:
        tokens = tokens[:-(len(tokens) % B)]
    assert len(tokens) >= (T + 1) * B, f"Expected {(T + 1) * B} tokens, got {len(tokens)}"

    buf = torch.tensor(tokens[:B * T + 1])
    x = buf[:-1].view(B, T).to(device)
    y = buf[1:].view(B, T).to(device)

    return x, y


if __name__ == '__main__':
    with open(TRAINING_DATA_PATH, "r") as f:
        text = f.read()
    data = text

    tokens = tokenize(data)

    if NUM_BATCHES > 1:
        tokens = tokens[:-(len(tokens) % NUM_BATCHES)]
    assert len(tokens) >= (NUM_TOKENS + 1) * NUM_BATCHES, f"Expected {(NUM_TOKENS + 1) * NUM_BATCHES} tokens, got {len(tokens)}"

    buf = torch.tensor(tokens[:NUM_BATCHES * NUM_TOKENS + 1])
    x = buf[:-1].view(NUM_BATCHES, NUM_TOKENS)
    y = buf[1:].view(NUM_BATCHES, NUM_TOKENS)
