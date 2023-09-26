def f(x):
    return x*x - 1

def derivative(f, x, delta):
    return (f(x + delta) - f(x)) / delta

x = 1
deltas = [10**-i for i in range(4, 15, 2)]

for delta in deltas:
    print("For δ = {}, the derivative of f at x = {} is {}".format(delta, x, derivative(f, x, delta)))