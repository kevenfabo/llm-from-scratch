import torch 

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
    
# instantiate a model object 
model = NeuralNetwork(50, 3)

print(model)

# Number of trainable parameters 
nums_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print(f"Total number of trainable model parameters: {nums_params}")

print(model.layers[0].weight)
print(model.layers[0].weight.shape)
print(model.layers[0].bias)
print(model.layers[0].bias.shape)

# Use random numbers as initial values
torch.manual_seed(123)
model = NeuralNetwork(50, 3)
print(model.layers[0].weight)

# Forward pass
torch.manual_seed(123)
X = torch.rand(
    [1, 50]
)
out = model(X)
print(out)
print(out.shape)

# If the model is only used for inference, constructing the computational graph
# for backpropagation is not necessary.

with torch.no_grad():
    out = model(X)
print(out)
print(out.shape)

with torch.no_grad():
    out = torch.softmax(model(X), dim=1)
    
print(out)