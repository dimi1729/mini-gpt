import tiktoken
import torch

def tokenize(sentence: str) -> torch.Tensor:
    '''Make my own custom tokenizer but for now use gpt one'''
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(sentence)
    return torch.tensor(tokens, dtype=torch.long)
