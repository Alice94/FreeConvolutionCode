import spectral_measure as sm
import numpy
import matplotlib.pyplot as plt
import freeconvolution
import scipy
from random import choices

# Example 1: Free additive convolution of semicircle and Marchenko-Pastur with parameter 0.5

# Input measures
mu1 = sm.spectral_measure()
mu2 = sm.spectral_measure()
mu1.set_semicircle(1)
mu2.set_marchenko_pastur(0.5)

# Parameters
N1 = 400 # quadrature points
N2 = 400
N = 400
m = 20 # number of coefficients to consider in the power series expansion (last step of the algorithm)
epsilon = 0.05

# Free additive convolution
[a_sum, b_sum, t, approx_mu, muplus] = freeconvolution.free_additive_convolution(mu1, mu2, N1, N2, N, m, epsilon, 1)

# Random matrices corresponding to the input measures
matrix_size = 1000
A1 = mu1.get_random_matrix(matrix_size)
A2 = mu2.get_random_matrix(matrix_size)
A = A1 + A2
eigenvalues = scipy.linalg.eigvals(A)

# Plot, in the (4,3) figure, the histogram of the eigenvalues of A compared 
# with the measure mu = mu1 x mu2 computed numerically
plt.subplot(4, 3, 10)
bin_number = int(50)
plt.plot(t, approx_mu * matrix_size * (b_sum - a_sum) / bin_number)
plt.hist(eigenvalues, bins = bin_number)
figure = plt.gcf()
figure.set_size_inches(16, 12)
plt.show()
plt.figure()
mu1.plot('b', 'mu1')
mu2.plot('g', 'mu2')
muplus.plot('r', 'sum')
plt.show()

del(mu1)
del(mu2)

########################################################################

# Example 2: Free additive convolution of semicircle with "truncated semicircle" 
# (in the sense that all the negative eigenvalues are sent to zero)

# Input measures
mu1 = sm.spectral_measure()
mu2 = sm.spectral_measure()
mu1.set_semicircle(1)
mu2.set_truncated_semicircle(1)

# Parameters
N1 = 400 # quadrature points
N2 = 4000
N = 4000
m = 20 # number of coefficients to consider in the power series expansion (last step of the algorithm)
epsilon = 0.02

# Free additive convolution
[a_sum, b_sum, t, approx_mu] = freeconvolution.free_additive_convolution(mu1, mu2, N1, N2, N, m, epsilon, 1)

# Random matrices corresponding to the input measures
matrix_size = 1000
A1 = mu1.get_random_matrix(matrix_size)
A2 = mu2.get_random_matrix(matrix_size)
A = A1 + A2
eigenvalues = scipy.linalg.eigvals(A)

# Plot, in the (4,3) figure, the histogram of the eigenvalues of A compared 
# with the measure mu = mu1 x mu2 computed numerically
plt.subplot(4, 3, 10)
bin_number = int(50)
plt.plot(t, approx_mu * matrix_size * (b_sum - a_sum) / bin_number)
plt.hist(eigenvalues, bins = bin_number)
figure = plt.gcf()
figure.set_size_inches(16, 12)
plt.savefig("truncated_semicircle_sum.pdf", dpi=10000)
plt.show()

