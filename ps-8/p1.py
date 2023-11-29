import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Load the waveform from a text file
waveform = np.loadtxt('piano.txt')

# Plot the waveform
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(waveform)
plt.title('Waveform')

# Compute the FFT of the waveform
fft_coeffs = fft(waveform)

# Get the magnitudes of the first 10,000 coefficients
magnitudes = np.abs(fft_coeffs[:10000])

# Plot the magnitudes
plt.subplot(2, 1, 2)
plt.plot(magnitudes)
plt.title('FFT Coefficients')

# Display the plots
plt.tight_layout()
plt.show()


