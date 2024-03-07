import numpy as np
import time
import torch


# Функция f
def f(x1, x2):
    return (4 - x1)**2 + (x1 - x2**2)**2


def gradient(x):
    x1, x2 = x
    return np.array([(-8+4*x1-2*(x2**2)), (-4*x1*x2+4*(x2**3))])


def hessian(x):
    x1, x2 = x
    return np.array([[4, -4*x2], [-4*x2, (-4*x1+12*(x2**2))]])


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


# Метод перебора
def iteration_search(x_range, y_range, epsilon):
    x1_min, x1_max = x_range
    x2_min, x2_max = y_range
    min_val = float('inf')
    min_point = None

    start_time = time.time()

    for x1 in np.arange(x1_min, x1_max, epsilon):
        for x2 in np.arange(x2_min, x2_max, epsilon):
            current_val = f(x1, x2)
            if current_val < min_val:
                min_val = current_val
                min_point = (x1, x2)

    end_time = time.time()
    search_time = end_time - start_time

    return min_point, min_val, search_time


# Метод Ньютона
def newton_method(x0, epsilon):

    x = np.array(x0)
    x_torch = torch.tensor(x, requires_grad=True, dtype=torch.float64)
    min_val = float('inf')
    start_time = time.time()
    while True:
        gradient_value = gradient_torch(x_torch).detach().numpy()
        hessian_value = hessian_torch(x_torch).detach().numpy()
        # gradient_value = gradient(x)
        # hessian_value = hessian(x)
        delta_x = np.linalg.inv(hessian_value) @ gradient_value
        x_new = x - delta_x
        if np.linalg.norm(delta_x) < epsilon:
            min_val = f(*x_new)
            break
        x = x_new
        x_torch.data = torch.tensor(x, requires_grad=True, dtype=torch.float64)

    end_time = time.time()
    search_time = end_time - start_time

    return x_new, min_val, search_time


x_range = (0, 6)
y_range = (0, 3)
epsilon = 0.01
initial_guess = [1, 1]

iteration_result = iteration_search(x_range, y_range, epsilon)
newton_result = newton_method(initial_guess, epsilon)


print("Метод перебора:")
print("Точка минимума:", iteration_result[0])
print("Минимальное значение функции:", iteration_result[1])
print("Время выполнения (сек):", iteration_result[2])

print("\nМетод Ньютона:")
print("Точка минимума:", newton_result[0])
print("Минимальное значение функции:", newton_result[1])
print("Время выполнения (сек):", newton_result[2])