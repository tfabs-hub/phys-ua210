import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

waveform = np.loadtxt('trumpet.txt')
print(len(waveform))


plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(waveform)
plt.title('Waveform')

fft_coeffs = fft(waveform)

magnitudes = np.abs(fft_coeffs[:10000])

plt.subplot(2, 1, 2)
plt.plot(magnitudes)
plt.title('FFT Coefficients')

plt.tight_layout()
plt.show()
