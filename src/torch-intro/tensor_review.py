import torch 

# Creating tensors
tensor0d = torch.tensor(1) # scalar 
tensor1d = torch.tensor([1, 2, 3]) # one-dimensional tensor
tensor2d = torch.tensor(
    [
        [1, 2],
        [3, 4]
    ]
) # two-dim tensor

tensor3d = torch.tensor(
    [
        [
            [1, 2],
            [3, 4]
        ],
        [
            [5, 6],
            [7, 8]
        ]
    ]
)

print(f"Tensor 3D has a shape: {tensor3d.shape}")
print(f"Tensor 3D is of type: {tensor3d.dtype}")


float_vec = torch.tensor([1.0, 3.0])
print(f"Data type of {float_vec}")
print(float_vec.dtype)

float_vec_quant = tensor1d.to(torch.float32)


# Pytorch operations 
tensor2d = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
)
print(tensor2d)
print(tensor2d.shape)
print(tensor2d.reshape(3, 2))
print(tensor2d.view(3, 2))
print(tensor2d.T)
print(tensor2d.matmul(tensor2d.T))
print(tensor2d @ tensor2d.T)