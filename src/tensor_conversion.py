import numpy as np
import torch

# Load normalized image
image = np.load("normalized_frames/frame_000000.npy")

print("NumPy Shape :", image.shape)

# Convert NumPy array to tensor
tensor = torch.tensor(image)

print("Tensor Shape :", tensor.shape)

# Change HWC -> CHW
tensor = tensor.permute(2, 0, 1)

print("After permute :", tensor.shape)

# Add batch dimension
tensor = tensor.unsqueeze(0)

print("After unsqueeze :", tensor.shape)

print("Tensor Type :", type(tensor))
print("Tensor Datatype :", tensor.dtype)