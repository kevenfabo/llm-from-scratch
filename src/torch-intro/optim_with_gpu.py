
# Pytorch on GPU devices 
import torch

import torch 
import torch.nn.functional as F
from data_loaders import train_loader, X_train, y_train



tensor_1 = torch.tensor([1., 2., 3.])
tensor_2 = torch.tensor([4., 5., 6.])
print(tensor_1 + tensor_2)

# transfer tensors to apple silicon
tensor_1 = tensor_1.to("mps")
tensor_2 = tensor_2.to("mps")
print(tensor_1 + tensor_2)


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(20, num_outputs),
        )
        
    def forward(self, x):
        logits = self.layers(x)
        return logits


# Training loop on a GPU device
torch.manual_seed(123)
model = NeuralNetwork(
    num_inputs=2,
    num_outputs=2
)

# define the device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# transfer the model to the device
model = model.to(device)

# define an optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.5
)

number_of_epochs = 3

for epoch in range(number_of_epochs):
    
    model.train() # set the model to training mode
    
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        # transfer the data to the device
        features, targets = features.to(device), targets.to(device)
        logits = model(features)
        
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # update the weights
        
        print(f"Epoch: {epoch+1:03d}/{number_of_epochs:03d}"
              f" | Batch: {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train/Val Loss: {loss:.2f}")