import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("var_5.dat", delimiter=",")
# data = np.loadtxt("var_10.dat", delimiter=",")

t_values = data[0]
y_values = data[1]


def phi(x, t):
    return x[3]*t**3 + x[2]*t**2 + x[1]*t + x[0]


def f(x, t, y):
    return phi(x, t) - y


def jacobian(x, t):
    J = np.zeros((len(t), len(x)))
    J[:, 0] = 1
    J[:, 1] = t
    J[:, 2] = t**2
    J[:, 3] = t**3
    return J


# Метод Гаусса-Ньютона
def gauss_newton(t, y, x0, epsilon):
    x = x0
    J = jacobian(x, t)
    J_tr = np.transpose(J)
    while True:
        delta_x = np.dot(np.dot(np.linalg.inv(np.dot(J_tr, J)), J_tr), f(x, t, y))
        x = x - delta_x
        if np.linalg.norm(delta_x) < epsilon:
            break
    return x


x0 = np.array([-1, 15, -1, 0.5])
# x0 = np.array([1, 0, 1, 0])
epsilon = 0.001
x = [-6, 11, -6, 1]
# x = [5, -7, 1, 1]
x_optimal = gauss_newton(t_values, y_values, x0, epsilon)
print(x_optimal)

J = jacobian(x_optimal, t_values)
print("Число обусловленностей Якобиана:", np.linalg.cond(J))

t_plot = np.linspace(min(t_values), max(t_values), 101)
phi_optimal = phi(x_optimal, t_plot)
phi_x = phi(np.array(x), t_plot)
print(sum((phi_optimal-y_values)**2))
print(sum((phi_x-y_values)**2))

plt.plot(t_plot, phi_optimal, label=f"Модельная функция для x*={x_optimal}")
plt.plot(t_plot, phi_x, label='Модельная функция для x=[-6, 11, -6, 1]')
plt.scatter(t_values, y_values, color='red', label='Данные')
plt.xlabel('t')
plt.ylabel('y')
plt.title('График модельной функции')
plt.legend()
plt.grid(True)
plt.show()