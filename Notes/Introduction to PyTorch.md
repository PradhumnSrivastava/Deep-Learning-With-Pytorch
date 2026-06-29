

---

# What is PyTorch?

PyTorch is an **open-source Deep Learning framework** developed by Facebook's AI Research Lab (**FAIR**).

It is mainly used for:

- Deep Learning
- Computer Vision
- Natural Language Processing (NLP)
- Reinforcement Learning
- Research and Production

---

# Why PyTorch?

## Advantages

- Easy to learn and Pythonic syntax.
- Dynamic Computation Graph.
- GPU acceleration support.
- Large community support.
- Flexible and research-friendly.
- Seamless integration with NumPy.

---

# Installation

## Using pip

```bash
pip install torch torchvision torchaudio
```

Check installation:

```python
import torch

print(torch.__version__)
```

---

# Core Building Block: Tensor

A **Tensor** is the fundamental data structure in PyTorch.

> Tensor = Generalization of Scalars, Vectors, and Matrices.

---

# Tensor Dimensions

| Object | Dimension |
|---------|-----------|
| Scalar | 0D |
| Vector | 1D |
| Matrix | 2D |
| Tensor | nD |

Examples:

```python
# Scalar (0D)
scalar = torch.tensor(5)

# Vector (1D)
vector = torch.tensor([1,2,3])

# Matrix (2D)
matrix = torch.tensor([[1,2],
                       [3,4]])

# Tensor (3D)
tensor = torch.tensor([
    [[1,2],[3,4]],
    [[5,6],[7,8]]
])
```

---

# Creating Tensors

## From Python Lists

```python
import torch

x = torch.tensor([1,2,3,4])

print(x)
```

---

## Using zeros()

Creates tensor filled with zeros.

```python
x = torch.zeros((2,3))

print(x)
```

Output:

```python
tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

---

## Using ones()

```python
x = torch.ones((3,4))

print(x)
```

---

## Random Tensor

```python
x = torch.rand((2,3))

print(x)
```

Random values lie between 0 and 1.

---

## Integer Random Tensor

```python
x = torch.randint(low=0,
                  high=10,
                  size=(2,3))

print(x)
```

---

# Tensor Properties

```python
x = torch.rand((3,4))

print(x.shape)
print(x.dtype)
print(x.device)
```

Output:

```python
torch.Size([3,4])
torch.float32
cpu
```

---

# Tensor Data Types

Common datatypes:

| Data Type | Description |
|------------|------------|
| torch.float32 | Float |
| torch.float64 | Double |
| torch.int32 | Integer |
| torch.int64 | Long Integer |
| torch.bool | Boolean |

Example:

```python
x = torch.tensor([1,2,3],
                 dtype=torch.float32)

print(x.dtype)
```

---

# Tensor Shape

```python
x = torch.rand((2,3,4))

print(x.shape)
```

Output:

```python
torch.Size([2,3,4])
```

Meaning:

- 2 blocks
- 3 rows
- 4 columns

---

# Tensor Indexing and Slicing

```python
x = torch.tensor([[1,2,3],
                  [4,5,6]])

print(x[0])
print(x[:,1])
print(x[0,2])
```

Output:

```python
tensor([1,2,3])

tensor([2,5])

tensor(3)
```

---

# Tensor Operations

## Addition

```python
a = torch.tensor([1,2,3])
b = torch.tensor([4,5,6])

c = a + b

print(c)
```

---

## Subtraction

```python
c = a - b
```

---

## Multiplication

```python
c = a * b
```

Element-wise multiplication.

---

## Division

```python
c = a / b
```

---

# Matrix Multiplication

```python
A = torch.tensor([[1,2],
                  [3,4]])

B = torch.tensor([[5,6],
                  [7,8]])

C = torch.matmul(A,B)

print(C)
```

Output:

```python
tensor([[19,22],
        [43,50]])
```

Shortcut:

```python
C = A @ B
```

---

# Tensor Reshaping

## reshape()

```python
x = torch.arange(12)

x = x.reshape(3,4)

print(x)
```

---

## view()

```python
x = torch.arange(12)

x = x.view(3,4)

print(x)
```

---

# Flatten Tensor

Converts multidimensional tensor into 1D.

```python
x = torch.tensor([[1,2],
                  [3,4]])

print(torch.flatten(x))
```

Output:

```python
tensor([1,2,3,4])
```

---

# Unsqueeze

Adds new dimension.

```python
x = torch.tensor([1,2,3])

print(x.shape)

x = x.unsqueeze(0)

print(x.shape)
```

Output:

```python
torch.Size([3])

torch.Size([1,3])
```

---

# Squeeze

Removes dimensions of size 1.

```python
x = torch.rand((1,3,1))

print(x.shape)

x = x.squeeze()

print(x.shape)
```

---

# NumPy ↔ Tensor Conversion

## NumPy to Tensor

```python
import numpy as np

arr = np.array([1,2,3])

tensor = torch.from_numpy(arr)

print(tensor)
```

---

## Tensor to NumPy

```python
tensor = torch.tensor([1,2,3])

arr = tensor.numpy()

print(arr)
```

---

# GPU Support

Check GPU availability.

```python
print(torch.cuda.is_available())
```

---

# Moving Tensor to GPU

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.tensor([1,2,3])

x = x.to(device)

print(x.device)
```

---

# Automatic Differentiation (Autograd)

PyTorch automatically computes gradients.

---

## requires_grad

```python
x = torch.tensor(2.0,
                 requires_grad=True)

y = x ** 2

y.backward()

print(x.grad)
```

Output:

```python
tensor(4.)
```

Explanation:

Given:

$$
y = x^2
$$

Derivative:

$$
\frac{dy}{dx}=2x
$$

At:

$$
x=2
$$

Gradient:

$$
2(2)=4
$$

---

# Computational Graph

Example:

```python
x = torch.tensor(3.0,
                 requires_grad=True)

y = x + 2

z = y * y

z.backward()

print(x.grad)
```

PyTorch internally creates:

```text
x → y → z
```

and applies backpropagation automatically.

---

# Basic Neural Network Example

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2,4),
    nn.ReLU(),
    nn.Linear(4,1)
)

x = torch.tensor([[1.0,2.0]])

output = model(x)

print(output)
```

---

# Most Important Modules

| Module | Purpose |
|---------|---------|
| torch | Core library |
| torch.nn | Neural network layers |
| torch.optim | Optimizers |
| torch.autograd | Automatic differentiation |
| torch.utils.data | Dataset and DataLoader |
| torchvision | Computer Vision utilities |

---

# Typical Deep Learning Workflow in PyTorch

```text
1. Prepare Dataset
        ↓
2. Create DataLoader
        ↓
3. Build Neural Network
        ↓
4. Define Loss Function
        ↓
5. Define Optimizer
        ↓
6. Forward Pass
        ↓
7. Compute Loss
        ↓
8. Backpropagation
        ↓
9. Update Weights
        ↓
10. Repeat for Epochs
```

---

# Quick Revision

## Tensor Creation

```python
torch.tensor()
torch.zeros()
torch.ones()
torch.rand()
torch.randint()
torch.arange()
```

## Shape Operations

```python
reshape()
view()
squeeze()
unsqueeze()
flatten()
```

## Math Operations

```python
+
-
*
/
matmul()
@
```

## GPU

```python
torch.cuda.is_available()

tensor.to(device)
```

## Autograd

```python
requires_grad=True

loss.backward()
```

---

# Interview Questions

### Q1: What is PyTorch?

A Deep Learning framework developed by Facebook, widely used for research and production.

---

### Q2: What is a Tensor?

A multidimensional array used as the basic data structure in PyTorch.

---

### Q3: Difference between NumPy array and Tensor?

Tensor supports GPU acceleration and automatic differentiation.

---

### Q4: What is Autograd?

PyTorch's automatic differentiation engine used during backpropagation.

---

### Q5: What is Dynamic Computational Graph?

The graph is created during runtime, making debugging easier and code more flexible.

---

# Summary

> PyTorch = Tensor + Autograd + Neural Networks + GPU Support
