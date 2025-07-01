import tiktoken
import torch

ENCODING = tiktoken.get_encoding('gpt2')

def tokenize(sentence: str) -> torch.Tensor:
    '''Make my own custom tokenizer but for now use gpt one'''
    tokens = ENCODING.encode(sentence)
    return torch.tensor(tokens, dtype=torch.long)

def decode(tokens: torch.Tensor) -> str:
    '''Decode tokens back into a string'''
    return ENCODING.decode(tokens.tolist())
