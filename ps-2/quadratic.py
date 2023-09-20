
import numpy as np

def find_roots(a, b, c):
    d = (b**2) - (4*a*c)

    sol1 = (-b - np.sqrt(d)) / (2*a)
    sol2 = (-b + np.sqrt(d)) / (2*a)

    alt_sol1 = 2*c / (-b - np.sqrt(d))
    alt_sol2 = 2*c / (-b + np.sqrt(d))

    if abs(a*sol1**2 + b*sol1 + c) < abs(a*alt_sol1**2 + b*alt_sol1 + c):
        accurate_sol1 = sol1
    else:
        accurate_sol1 = alt_sol1

    if abs(a*sol2**2 + b*sol2 + c) < abs(a*alt_sol2**2 + b*alt_sol2 + c):
        accurate_sol2 = sol2
    else:
        accurate_sol2 = alt_sol2

    return accurate_sol1, accurate_sol2

a = 0.001
b = 1000
c = 0.001

sol1, sol2 = find_roots(a, b, c)

print("The most accurate solutions are {0} and {1}".format(sol1,sol2))