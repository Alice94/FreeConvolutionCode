import spectral_measure as sm
import numpy
import matplotlib.pyplot as plt
import freeconvolution
import scipy
from random import choices

# Set up parameters
do_plots = 1
epsilon = 0.05
matrix_size = 5000 # then it should be 5000
bin_number = int(50)
  
mu1 = sm.spectral_measure()
mu2 = sm.spectral_measure()

N1 = 400
N2 = 400
N = 400
m = 20
param = 0.5
mu2.set_semicircle(1)
mu1.set_marchenko_pastur(param)


name = "explain1a.eps"
[a_sum, b_sum, t, approx_mu, muplus] = freeconvolution.free_additive_convolution(mu1, mu2, N1, N2, N, m, epsilon, 2)
figure = plt.gcf()
figure.set_size_inches(16, 4)
plt.savefig(name, dpi=10000)
plt.show()

del(mu1)
del(mu2)
mu1 = sm.spectral_measure()
mu2 = sm.spectral_measure()
mu2.set_semicircle(1)
mu1.set_marchenko_pastur(param)


name = "explain1b.eps"
[a_sum, b_sum, t, approx_mu, muplus] = freeconvolution.free_additive_convolution(mu2, mu1, N1, N2, N, m, epsilon, 2)
figure = plt.gcf()
figure.set_size_inches(16, 4)
plt.savefig(name, dpi=10000)
plt.show()

del(mu1)
del(mu2)
mu1 = sm.spectral_measure()
mu2 = sm.spectral_measure()
mu2.set_semicircle(1)
mu1.set_marchenko_pastur(param)

name = "explain2.eps"
[a_sum, b_sum, t, approx_mu, muplus] = freeconvolution.free_additive_convolution(mu2, mu1, N1, N2, N, m, epsilon, 3)
figure = plt.gcf()
figure.set_size_inches(16, 4)
plt.savefig(name, dpi=10000)
plt.show()

del(mu1)
del(mu2)
mu1 = sm.spectral_measure()
mu2 = sm.spectral_measure()
mu2.set_semicircle(1)
mu1.set_marchenko_pastur(param)

name = "explain3.eps"
[a_sum, b_sum, t, approx_mu, muplus] = freeconvolution.free_additive_convolution(mu1, mu2, N1, N2, N, m, epsilon, 4)
figure = plt.gcf()
figure.set_size_inches(10, 4)

A1 = mu1.get_random_matrix(matrix_size)
A2 = mu2.get_random_matrix(matrix_size)
A = A1 + A2
eigenvalues = scipy.linalg.eigvals(A)

plt.subplot(1, 2, 1)
(counts, bins) = numpy.histogram(eigenvalues, bins=bin_number)
plt.hist(bins[:-1], bins, weights=bin_number / (matrix_size * (muplus.b - muplus.a)) * counts)
plt.plot(t, approx_mu, 'r')

plt.savefig(name, dpi=10000)
plt.show()
