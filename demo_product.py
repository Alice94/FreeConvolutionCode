import spectral_measure as sm
import numpy
import matplotlib.pyplot as plt
import freeconvolution
import scipy
from random import choices

# Example: Free multiplicative convolution of shifted semicircle and Marchenko-Pastur with parameter 0.2

# Input measures
mu1 = sm.spectral_measure()
mu2 = sm.spectral_measure()
mu1.set_shifted_semicircle(1)
mu2.set_marchenko_pastur(0.2)

# Parameters
N1 = 400 # quadrature points
N2 = 400
N = 400
m = 20 # number of coefficients to consider in the power series expansion (last step of the algorithm)
epsilon = 0.05

# Free multiplicative convolution
[a_prod, b_prod, t, approx_mu, muprod] = freeconvolution.free_multiplicative_convolution(mu1, mu2, N1, N2, N, m, epsilon, 1)

# Random matrices corresponding to the input measures
matrix_size = 1000
A1 = mu1.get_random_matrix(matrix_size)
A2 = mu2.get_random_matrix(matrix_size)
A = numpy.matmul(A1, A2)
eigenvalues = scipy.linalg.eigvals(A)

# Plot, in the (4,3) figure, the histogram of the eigenvalues of A compared 
# with the measure mu = mu1 x mu2 computed numerically
plt.subplot(4, 3, 10)
bin_number = int(50)
plt.plot(t, approx_mu * matrix_size * (b_prod - a_prod) / bin_number)
plt.hist(eigenvalues, bins = bin_number)
figure = plt.gcf()
figure.set_size_inches(16, 12)
plt.show()

del(mu1)
del(mu2)

##################################

# I also tried with a "truncated" distribution, visually it is sort-of-ok but it's not working well,
# in particular it cannot find the left extremum of the support accurately. 
# Maybe this is fixable by playing with the parameters

# Input measures
mu1 = sm.spectral_measure()
mu2 = sm.spectral_measure()
mu1.set_shifted_semicircle(1)
mu2.set_truncated_semicircle_2(1)

# Parameters
N1 = 400 # quadrature points
N2 = 4000
N = 4000
m = 20 # number of coefficients to consider in the power series expansion (last step of the algorithm)
epsilon = 0.02

# Random matrices corresponding to the input measures
matrix_size = 1000
A1 = mu1.get_random_matrix(matrix_size)
A2 = mu2.get_random_matrix(matrix_size)
A = numpy.matmul(A1, A2)
eigenvalues = scipy.linalg.eigvals(A)

# Free multiplicative convolution
[a_prod, b_prod, t, approx_mu, muprod] = freeconvolution.free_multiplicative_convolution(mu1, mu2, N1, N2, N, m, epsilon, 1)

# Plot, in the (4,3) figure, the histogram of the eigenvalues of A compared 
# with the measure mu = mu1 x mu2 computed numerically
plt.subplot(4, 3, 10)
bin_number = int(50)
plt.plot(t, approx_mu * matrix_size * (b_prod - a_prod) / bin_number)
plt.hist(eigenvalues, bins = bin_number)
figure = plt.gcf()
figure.set_size_inches(16, 12)
plt.savefig("truncated_semicircle_prod.pdf", dpi=10000)
plt.show()
