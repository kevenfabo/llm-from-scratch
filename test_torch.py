

import torch
x = torch.rand(5, 3)
print(x)


from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

import torch
print(torch.__version__)


