import torch 
import torch.nn.functional as F
from data_loaders import train_loader, X_train, y_train

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
    
torch.manual_seed(123)
model = NeuralNetwork(
    num_inputs=2,
    num_outputs=2
)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.5
)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    
    for batch_idx, (features, targets) in enumerate(train_loader):
        logits = model(features)
        loss = F.cross_entropy(logits, targets)
        
        optimizer.zero_grad() # set the gradients to zero to prevent gradient accumulation
        loss.backward() # compute the gradients
        optimizer.step() # update the weights
        
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch: {batch_idx:03d}/{len(train_loader):03d}"
              f" | Loss: {loss:.2f}")
    
    model.eval()


# Number of trainable parameters
nums_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)
print(f"Total number of trainable model parameters: {nums_params}")

# Forward pass on the trained data
model.eval()
with torch.no_grad():
    outputs = model(X_train)
print(outputs)

# Get the predicted probabilities
torch.set_printoptions(sci_mode=False) # disable scientific notation
probs = torch.softmax(outputs, dim=1)
print(probs)

# Get the predicted labels
pred_labels = torch.argmax(probs, dim=1)
print(pred_labels)

# lets compare the predicted labels with the actual labels
print(pred_labels == y_train)

# number of correct predictions
accuracy = torch.sum(pred_labels == y_train) / y_train.shape[0]
print(f"Accuracy: {accuracy:.2f}")


def compute_accuracy(model, dataloader):
    
    model = model.eval() # set the model to evaluation mode
    correct = 0
    total_examples = 0
    
    for batch_idx, (features, targets) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)
            
        predictions = torch.argmax(logits, dim=1)
        correct += torch.sum(predictions == targets)
        total_examples += targets.shape[0]
        
    return correct / total_examples

print(f"Accuracy on training set: {compute_accuracy(model, train_loader):.2f}")


# save the model
torch.save(model.state_dict(), "model/model.pth")

# Use the saved model
saved_model = NeuralNetwork(
    num_inputs=2,
    num_outputs=2
)
saved_model.load_state_dict(
    torch.load(
        "model/model.pth"
    )
)

# Forward pass
saved_model.eval()
with torch.no_grad():
    outputs = saved_model(X_train)
pred_labels = torch.argmax(
    torch.softmax(outputs, dim=1),
    dim=1
)
print(pred_labels)

# predict labels without softmax
with torch.no_grad():
    outputs = saved_model(X_train)
pred_labels = torch.argmax(outputs, dim=1)
print("Label without softmax:", pred_labels)