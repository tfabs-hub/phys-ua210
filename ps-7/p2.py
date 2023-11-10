import jax.numpy as np
from jax import grad, hessian, jit
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('survey.csv')


def logistic(x, beta0, beta1):
    return 1 / (1 + np.exp(-beta0 - beta1 * x))

def neg_log_likelihood(beta, x, y):
    p = logistic(x, beta[0], beta[1])
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


grad_neg_log_likelihood = jit(grad(neg_log_likelihood, argnums=0))
hessian_neg_log_likelihood = jit(hessian(neg_log_likelihood, argnums=0))


age = df.iloc[:, 0].values
answer = df.iloc[:, 1].values


beta_init = np.array([0.0, 0.0])


res = minimize(neg_log_likelihood, beta_init, args=(age, answer), jac=grad_neg_log_likelihood, hess=hessian_neg_log_likelihood, method='Newton-CG')


print("Maximum likelihood estimates for beta:", res.x)


cov_matrix = np.linalg.inv(hessian_neg_log_likelihood(res.x, age, answer))
print("Covariance matrix:", cov_matrix)


plt.scatter(age, answer, label='Data')
plt.plot(age, logistic(age, res.x[0], res.x[1]), label='Logistic model')
plt.xlabel('Age')
plt.ylabel('Answer')
plt.legend()
plt.show()