import numpy as np
from scipy.optimize import minimize_scalar

def brent(f, a, b, tol=1e-5):
    
    golden = (3 - np.sqrt(5)) / 2
    
    
    x = w = v = (a + b) / 2
    fw = fv = fx = f(x)
    d = e = b - a
    while abs(x - (a + b) / 2) + (b - a) / 2 > 2 * tol:
        g = e
        e = d
        if x != w and x != v and w != v:
            f1 = (w - x) * (fw - fx)
            f2 = (v - x) * (fv - fx)
            f3 = (w - x) * f1
            f4 = (v - x) * f2
            f5 = f1 - f2
            t = x + (f3 * f4 * (w - v) + f2 * f3 * (v - x) - f1 * f4 * (w - x)) / (2 * f5)
            if a < t < b and abs(t - x) < g / 2:
                u = t
                fu = f(u)
                if fu <= fx:
                    if u < x:
                        b = x
                    else:
                        a = x
                    v, w, x = w, x, u
                    fv, fw, fx = fw, fx, fu
                else:
                    if u < x:
                        a = u
                    else:
                        b = u
                    if fu <= fw or w == x:
                        v, w = w, u
                        fv, fw = fw, fu
                    elif fu <= fv or v == x or v == w:
                        v = u
                        fv = fu
            else:
                if x < (a + b) / 2:
                    t = x + golden * (b - x)
                    e = b - x
                else:
                    t = x - golden * (x - a)
                    e = x - a
        else:
            if x < (a + b) / 2:
                t = x + golden * (b - x)
                e = b - x
            else:
                t = x - golden * (x - a)
                e = x - a
        if abs(t - x) < tol:
            u = t if t > x else t - tol
        else:
            u = t
        fu = f(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu
    return x

# Test function
f = lambda x: (x - 0.3)**2 * np.exp(x)

# Call the brent function
result = brent(f, -10, 10)
print("The minimum of the function occurs at x =", result)



res = minimize_scalar(f, method='brent')
print("The minimum of the function occurs at x =", res.x)