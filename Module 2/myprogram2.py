import torch 
import matplotlib.pyplot as plt

# Create tensors with requires_grad=True)
x = torch.tensor([2.0, 5.0], requires_grad=True)
y = torch.tensor([3.0, 7.0], requires_grad=True)

# Perform some operations 
z = x * y + y**2

z.retain_grad() # By default intermediate layer weight updation is not shown.

# Compute the gradients
z_sum = z.sum().backward()


print(f"Gradient of x: {x.grad}")
print(f"Gradient of y: {y.grad}")
print(f"Gradient of z: {z.grad}")
print(f"Result of the operation: z = {z.detach()}")


# 1.2 Gradient Computation Graph 

from torchviz import make_dot

# Visualize the computation graph
dot = make_dot(z, params={"x": x, "y": y, "z" :z})
dot.render("grad_computation_graph", format="png")

img = plt.imread("grad_computation_graph.png")
plt.imshow(img)
plt.axis('off')
plt.show()

# 1.3 Detaching Tensors from Computation Graph

# Let's detach z from the computation graph
print("Before detaching z from computation: ", z.requires_grad)
z_det = z.detach()
print("After detaching z from computation: ", z_det.requires_grad)










