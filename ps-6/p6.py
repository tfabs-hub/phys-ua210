import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

hdu_list = fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data
hdu_list.close()

wavelength = 10**logwave

for i in range(5):
    plt.plot(wavelength, flux[i], label=f'Galaxy {i+1}')

plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Flux (10^-17 erg s^-1 cm^-2 A^-1)')
plt.legend()
plt.show()



norm_factors = np.zeros(flux.shape[0])


for i in range(flux.shape[0]):
    norm_factors[i] = np.sum(flux[i])
    flux[i] /= norm_factors[i]
    

mean_spectrum = np.mean(flux, axis=0)

residuals = np.zeros_like(flux)


for i in range(flux.shape[0]):
    residuals[i] = flux[i] - mean_spectrum
    
    

C = np.dot(residuals.T, residuals) / residuals.shape[0]
eigvals, eigvecs = np.linalg.eigh(C)
eigvecs = eigvecs[:, ::-1]


for i in range(5):
    plt.plot(wavelength, eigvecs[:, i], label=f'Eigenvector {i+1}')

plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Eigenvector')
plt.legend()
plt.show()


U, s, Vt = np.linalg.svd(residuals, full_matrices=False)
eigvecs_svd = Vt.T


for i in range(5):
    plt.plot(wavelength, eigvecs_svd[:, i], label=f'Eigenvector {i+1}')

plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('Eigenvector')
plt.legend()
plt.show()


dot_product = np.dot(Vt, eigvecs)


print(dot_product)

coefficients = np.dot(residuals, Vt.T)
coefficients[:, 5:] = 0

approx_spectra = mean_spectrum + np.dot(coefficients, Vt)

approx_spectra *= norm_factors[:, np.newaxis]

for i in range(5):
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, flux[i], label='Original')
    plt.plot(wavelength, approx_spectra[i], label='Approximate')
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Flux')
    plt.legend()
    plt.title(f'Galaxy {i+1}')
    plt.show()
    
    
c0 = coefficients[:, 0]
c1 = coefficients[:, 1]
c2 = coefficients[:, 2]

plt.figure(figsize=(10, 6))
plt.scatter(c0, c1)
plt.xlabel('c0')
plt.ylabel('c1')
plt.title('c0 vs c1')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(c0, c2)
plt.xlabel('c0')
plt.ylabel('c2')
plt.title('c0 vs c2')
plt.show()

sq_residuals = np.zeros(20)

for Nc in range(1, 21):
    coefficients = np.dot(residuals, Vt.T)

    coefficients[:, Nc:] = 0

    approx_spectra = mean_spectrum + np.dot(coefficients, Vt)

    residuals_approx = flux - approx_spectra

    sq_residuals[Nc-1] = np.mean((residuals_approx / (flux + 1e-8))**2)

print(sq_residuals)



