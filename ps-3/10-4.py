

import numpy as np
import matplotlib.pyplot as plt

N = 1000
T_half = 3.053
T_half_seconds = T_half * 60
lambda_ = np.log(2) / T_half_seconds



uniformNumbers = np.random.uniform(size=N)
decayTimes = -np.log(1 - uniformNumbers) / lambda_
sortedDecayTimes = np.sort(decayTimes)
notDecayed = np.arange(N, 0, -1)


plt.plot(sortedDecayTimes, notDecayed)
plt.xlabel('Time (s)')
plt.ylabel('Number of Atoms Not Decayed')
plt.title('Decay of Atoms Over Time')
plt.grid(True)
plt.show()