import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**4 - 2*x + 1

def trapezoid(func, a, b, N):
    h = (b - a) / N
    s = 0.5 * (func(a) + func(b))
    for i in range(1, N):
        s += func(a + i*h)
    return h * s

I1 = trapezoid(f, 0, 2, 10)

I2 = trapezoid(f, 0, 2, 20)


#error as accoriding to 5.28
e2 = (1/3) * (I2 - I1)


trueValue = 4.4


directError = np.abs(trueValue - I2)

print(f"Integral with M=10 slices: {I1}")
print(f"Integral with N=20 slices: {I2}")
print(f"Estimated error: {e2}")
print(f"Direct error: {directError}")
