# The goal of this script is ways to create and use datasets and data loaders in PyTorch.
import torch 
from torch.utils.data import Dataset, DataLoader

X_train = torch.tensor(
    [
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ]
)
y_train = torch.tensor(
    [0, 0, 0, 1, 1]
)
X_test = torch.tensor(
    [
        [-0.8, 2.8],
        [2.6, -1.6]   
    ]
)
y_test = torch.tensor(
    [0, 1]
)

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y
        
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
    def __len__(self):
        return self.labels.shape[0]
    
train_dataset = ToyDataset(X_train, y_train)
test_dataset = ToyDataset(X_test, y_test)

# print(len(train_dataset))

torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=2,
    shuffle=True, # shuffle the data
    num_workers=0,
    drop_last=True # drop the last batch if it is not complete
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=2,
    shuffle=False, # not necessary to shuffle the test data
    num_workers=0,
)

if __name__ == "__main__":
    for idx, (x, y) in enumerate(train_loader):
        print(f"Batch: {idx + 1}:", x, y)
    