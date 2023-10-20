import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



signal = pd.read_csv('signal.dat',sep="|",usecols=[1,2])
signal=signal.to_numpy()


t=np.transpose(signal)[0]
t=t-np.mean(t)
t=t/np.sqrt(np.sum(t**2))


sig=np.transpose(signal)[1]

#plt.scatter(t, sig)

'''def SVD(N):

    A = np.zeros((len(t), N))
    
    for i in range(N):
       
        A[:, i] = t**i
      
    
    (U, S, Vh) = np.linalg.svd(A, full_matrices=False)
    
    print(np.min(S)/np.max(S))
    
    A_inv = Vh.transpose().dot(np.diag(1. / S)).dot(U.transpose())
    c = A_inv.dot(sig)
    
    SigM = A.dot(c)
    
    return SigM


G = 350

plt.plot(t, sig, '.')
plt.plot(t, SVD(G), '.')
plt.xlabel('Value')
plt.ylabel('Number')
plt.title('SVD Fit With 350th Order Polynomial')
#plt.title('Residuals for a third order polynomial')

Residuals = sig - SVD(G)
#plt.hist(Residuals,bins = 21)
'''


def fourier_series(t, *a):
    ret = a[0] * np.cos(t / (2 * np.pi))
    for deg in range(1, len(a)):
        ret += a[deg] * np.cos(deg * t / (2 * np.pi)) + a[deg] * np.sin(deg * t / (2 * np.pi))
    return ret


t = np.linspace(0, 4*np.pi, num=1000)
sig = np.sin(t) + np.random.normal(size=1000) * 0.2

plt.plot(fourier_series(t, sig))

