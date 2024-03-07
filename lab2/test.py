import torch
import numpy as np
initial_guess = [1, 1]


def f_torch(x):
    return torch.sum((4 - x[0]) ** 2 + (x[0] - x[1] ** 2) ** 2)


def gradient_torch(x):
    return torch.autograd.grad(f_torch(x), x, create_graph=True)[0]


def hessian_torch(x):
    gradient = gradient_torch(x)
    hessian = []
    for g in gradient:
        hessian.append(torch.autograd.grad(g, x, retain_graph=True)[0])
    return torch.stack(hessian)


x_torch = torch.tensor(initial_guess, requires_grad=True, dtype=torch.float64)

num_iterations = 5

for _ in range(num_iterations):
    initial_guess[0]+=1
    initial_guess[1]+=1
    gradient_value = gradient_torch(x_torch)
    hessian_value = hessian_torch(x_torch)
    print(gradient_value.detach().numpy())
    print(hessian_value.detach().numpy())

gradient_value = gradient_torch(x_torch).detach().numpy()
hessian_value = hessian_torch(x_torch).detach().numpy()

print("Градиент:", gradient_value)
print("Гессиан:", hessian_value)