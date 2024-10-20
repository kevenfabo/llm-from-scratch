import torch

def get_pytorch_version():
    print(
        f"Pytorch version is: {torch.__version__}"
    )


if __name__ == "__main__":
    
    # check the pytorch version
    get_pytorch_version()
    
    # check if cuda is available
    print(
        f"CUDA available: {torch.cuda.is_available()}"
    )
    
    # check if Apple silicon chip available 
    print(
        f"Apple silicon chip available: {torch.backends.mps.is_available()}"
    )
    
