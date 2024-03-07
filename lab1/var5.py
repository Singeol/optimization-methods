import math
import numpy
from sympy import lambdify, symbols


def dichotomy_method(f, c, a, b, epsilon):
    n = 0
    while (b - a) / 2 > epsilon:
        x1 = (a + b - epsilon) / 2
        x2 = (a + b + epsilon) / 2
        if c * f(x1) < c * f(x2):
            b = x2
        else:
            a = x1
        n += 1
    result = (a + b) / 2
    return result, n


def main():
    x = symbols("x")
    f = lambdify(x, input("Введите функцию:\n"), "numpy")
    work = 'y'
    while work != 'n':
        a = float(input("Введите a: "))
        b = float(input("Введите b: "))
        sign = int(input("Введите какой экстремум ищем:\n"
                         "1 - минимум\n"
                         "-1 - максимум\n"))

        accuracy = float(input("Введите точность определения экстремума: "))
        result, iterations = dichotomy_method(f, sign, a, b, accuracy)
        print("---------------------------------------")
        print(f"Число итераций: {iterations}")
        print(f"Приближенное значение точки экстремума: {round(result, int(-math.log10(accuracy)))}")
        print(f"Значение функции в данной точке: {round(f(result), int(-math.log10(accuracy)))}")
        print("---------------------------------------\n")
        work = input("Для выхода введите - n, для продолжения работы - y: ")


if __name__ == "__main__":
    main()