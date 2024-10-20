import torch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from data_loaders import train_dataset, test_dataset
from torch.utils.data import DataLoader

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
    

def ddp_setup(rank, world_size):
    
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # initialize the process group
    init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    
    # set the current mps device to the current rank
    torch.cuda.set_device(rank)
    
    
def prepare_datasets(rank, world_size):
    
    # create the data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=2,
        sampler=DistributedSampler(
            dataset=train_dataset,
        ),
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=2,
        sampler=DistributedSampler(
            dataset=test_dataset,
        ),
        shuffle=False,
        pin_memory=False
    )
    
    return train_loader, test_loader

def main(rank, world_size, number_epochs):
    
    ddp_setup(rank, world_size)
    train_loader, test_loader = prepare_datasets(rank, world_size)
    
    model = NeuralNetwork(
        num_inputs=2,
        num_outputs=2
    )
    model = model.to(rank)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.5
    )
    model = DDP(model, device_ids=[rank])
    
    for epoch in range(number_epochs):
        pass # training loop