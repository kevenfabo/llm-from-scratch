import torch
import torch.nn.functional as F

# Let's implement a LR forward pass
y = torch.tensor([1.0]) # True label 
x1 = torch.tensor([1.1]) # Input feature 
w1 = torch.tensor([2.2]) # Weight parameter
b = torch.tensor([0.0]) # Bias unit
z = x1 * w1 + b # Net input 
a = torch.sigmoid(z) # Activation and output
loss = F.binary_cross_entropy(a, y)

print(f"Loss: {loss}")

