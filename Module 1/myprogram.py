import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

## 2.1 Construct tensor 
# Create a tensor with ones in a column
a = torch.ones(5)

# Print the tensor created
print(a)
# Create a tensor with zeros in a column 
b = torch.zeros(5)
print(b)

c = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print(c)

d = torch.zeros(3, 2)
print(d)

e = torch.ones(3, 2)
print(e)

f = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(f)

# Create a 3D tensor
g = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
print(g)

print(e.shape)

print(f.shape)

print(g.shape)

## 2.2 Access an element of tensor
# Get element at index 2
print(c[2])

# Get element at row 1, column 0
print(f[1, 0])

# Alternative access method
print(f[1][0])

# Access element in 3D tensor
print(g[1, 0, 0])
print(g[1][0][0])

# Get all elements
print(f[:])

# Get elements from index 1 to 2 (excluding element 3)
print(c[1:3])

# Get all elements till index 4 (exclusive)
print(c[:4])

# Get first row 
print(f[0, :])

# Get second column 
print(f[:, 1])

## 2.3 Specify data type of elements
int_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(int_tensor.dtype)

# Change one element to floating number
int_tensor = torch.tensor([[1, 2, 3], [4., 5, 6]])
print(int_tensor.dtype)
print(int_tensor)

# Override data type to int64
float_tensor = torch.tensor([[1, 2, 3], [4., 5, 6]])
int_tensor = float_tensor.type(torch.int64)
print(int_tensor.dtype)
print(int_tensor)

## 2.4 Convert tensor to/from NumPy array
# Convert tensor to array 
f_numpy = f.numpy()
print(f_numpy)

# Convert array to tensor 
h = np.array([[8, 7, 6, 5], [4, 3, 2, 1]])
h_tensor = torch.from_numpy(h)
print(h_tensor)

## 2.5 Perform arithmetic operations on tensors
# Create tensors
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor2 = torch.tensor([[-1, -2, -3], [4, -5, 6]])

# Add tensors
print(tensor1 + tensor2)
# Add using torch.add()
print(torch.add(tensor1, tensor2))

# Subtract tensors
print(tensor1 - tensor2)
# Subtract using torch.sub()
print(torch.sub(tensor1, tensor2))

# Multiply tensor by scalar
print(tensor1 * 2)

# Multiply tensors elementwise 
print(tensor1 * tensor2)

# Perform matrix multiplication
tensor3 = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(torch.mm(tensor1, tensor3))

# Divide tensor by scalar
print(tensor1 / 2)

# Elementwise division of tensors
print(tensor1 / tensor2)

## 2.6 Broadcasting
# Create two 1-dimensional tensors
a = torch.tensor([1, 2, 3])
b = torch.tensor([4])

# Add scalar to vector
result = a + b 
print("Result of Broadcasting:\n", result)

# Create tensors with shapes (1, 3) and (3, 1)
a = torch.tensor([[1, 2, 3]])
b = torch.tensor([[4], [5], [6]])

# Add tensors with different shapes
result = a + b
print("Shape:", result.shape)
print("\n")
print("Result of Broadcasting:\n", result)

## 2.7 Use CPU tensors only (no GPU)
# Set device to CPU explicitly
device = torch.device("cpu")

# Create tensor on CPU
tensor_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)

# Multiply tensor by 5
tensor_cpu = tensor_cpu * 5

# GPU code removed to avoid errors on machines without NVIDIA GPU
