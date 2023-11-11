import jax.numpy as jnp
from jax import grad, hessian, jit
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('survey.csv')

age = df.iloc[:, 0].values
answer = df.iloc[:, 1].values

def logistic(x, beta0, beta1):
    return 1 / (1 + jnp.exp(-(beta0 + beta1*x)))

def neg_log_likelihood(beta, x, y):
    prob = logistic(x, beta[0], beta[1])
    return -jnp.sum(y * jnp.log(prob + 1e-5) + (1 - y) * jnp.log(1 - prob + 1e-5))


grad_neg_log_likelihood = jit(grad(neg_log_likelihood))
hessian_neg_log_likelihood = jit(hessian(neg_log_likelihood))


beta_init = jnp.array([0.0, 0.0])


res = minimize(neg_log_likelihood, beta_init, args=(age, answer), method='Newton-CG', jac=grad_neg_log_likelihood, hess=hessian_neg_log_likelihood)

print("Maximum likelihood estimates:")
print("beta0 =", res.x[0])
print("beta1 =", res.x[1])

plt.figure(figsize=(10, 6))
plt.scatter(age, answer, label='Data')
plt.plot(jnp.sort(age), logistic(jnp.sort(age), res.x[0], res.x[1]), color='red', label='Logistic model')
plt.xlabel('Age')
plt.ylabel('Answer')
plt.legend()
plt.show()