import torch
import torch.nn.functional as F
from torch.autograd import grad

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
z = w1 * x1
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)

# manual grad calculation
grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

print(grad_L_b)
print(grad_L_w1)

# grad calculation using .backward()
loss.backward()
print(w1.grad)
print(b.grad)