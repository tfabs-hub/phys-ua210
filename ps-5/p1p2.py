import numpy as np


def integrand(z, a, c):
    x = z * c / (1 - z)
    return (np.exp((a-1)*(np.log(x)) -x))

def gamma_func(a):
    c = a - 1
    z = np.linspace(0.01, .99, 1000)
    y = integrand(z, a, c)
    dz = c/(z-1)**2
    result = np.trapz((y)*dz, z)
    return result

print(gamma_func(3/2))
print(gamma_func(2))
print(gamma_func(3))
print(gamma_func(4))
print(gamma_func(5))
print(gamma_func(6))

