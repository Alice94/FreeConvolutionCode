import spectral_measure as sm
import numpy
import matplotlib.pyplot as plt
import freeconvolution
import scipy
from random import choices

# Set up parameters
do_plots = 1
epsilon = 0.05
matrix_size = 5000
bin_number = int(100)

for n_example in range(0, 5):
  
  mu1 = sm.spectral_measure()
  mu2 = sm.spectral_measure()
  
  if (n_example == 0):
    param = 0.1
    # ~ param2 = 0.2
    a2 = 1
    b2 = 3
    mu1.set_marchenko_pastur(param)
    mu2.set_uniform(a2, b2)
    # ~ mu2.set_marchenko_pastur(param2)
    N1 = 400
    N2 = 400
    N = 4000
    m = 20
    name = "MPxunif.eps"
    A1 = mu1.get_random_matrix(matrix_size)
    A2 = mu2.get_random_matrix(matrix_size)
    A = numpy.matmul(A1, A2)
    eigenvalues = scipy.linalg.eigvals(A)
    print(numpy.min(eigenvalues), numpy.max(eigenvalues))
    
  elif (n_example == 1):
    N1 = 4000
    N2 = 4000
    N = 4000
    m = 20
    param = 0.2
    mu1.set_shifted_semicircle(1)
    mu2.set_marchenko_pastur(param)
    name = "SemicirclexMP0.2.eps"
    A1 = mu1.get_random_matrix(matrix_size)
    A2 = mu2.get_random_matrix(matrix_size)
    A = numpy.matmul(A1, A2)
    eigenvalues = scipy.linalg.eigvals(A)
    epsilon = 0.05
  
  elif (n_example == 2):
    N1 = 400
    N2 = 400
    N = 400
    m = 20
    param = 1
    mu1.set_shifted_semicircle(1)
    mu2.set_shifted_semicircle(1)
    name = "Semicirclexsemicircle.eps"
    A1 = mu1.get_random_matrix(matrix_size)
    A2 = mu2.get_random_matrix(matrix_size)
    A = numpy.matmul(A1, A2)
    eigenvalues = scipy.linalg.eigvals(A)  
    
  elif (n_example == 3):
    N1 = 400
    N2 = 4000
    N = 4000
    m = 24
    mu1.set_shifted_semicircle(1)
    weights = numpy.array(numpy.ones(7))
    weights = weights / sum(weights)
    points = numpy.array(numpy.linspace(1,4,num=7))
    mu2.set_discrete(points, weights)
    name = "semicirclexdiscrete.eps"
    A1 = mu1.get_random_matrix(matrix_size)
    A2 = mu2.get_random_matrix(matrix_size)
    A = numpy.matmul(A1, A2)
    eigenvalues = scipy.linalg.eigvals(A)  
    epsilon = 0.08
  
  elif (n_example == 4):
    a1 = 1
    a2 = 1
    b1 = 2
    b2 = 3
    
    mu1.set_uniform(a1, b1)
    mu2.set_uniform(a2, b2)
    N1 = 4000
    N2 = 4000
    N = 4000
    m = 20
    name = "Unif1xUnif2.eps"
    A1 = mu1.get_random_matrix(matrix_size)
    A2 = mu2.get_random_matrix(matrix_size)
    A = numpy.matmul(A1, A2)
    eigenvalues = scipy.linalg.eigvals(A)
  
  

  [a_prod, b_prod, t, approx_mu, mutimes] = freeconvolution.free_multiplicative_convolution(mu1, mu2, N1, N2, N, m, epsilon, do_plots)
  plt.subplot(4, 3, 10)
  (counts, bins) = numpy.histogram(eigenvalues, bins=bin_number)
  plt.hist(bins[:-1], bins, weights=bin_number / (matrix_size * (mutimes.b - mutimes.a)) * counts)
  plt.plot(t, approx_mu, 'r')
  
  plt.subplot(4, 3, 11)
  plt.plot(t, abs(approx_mu), 'r')
    
  figure = plt.gcf() # get current figure
  figure.set_size_inches(16, 12)
  plt.savefig(name, dpi=10000)
  # ~ plt.show()
  
  del(mu1)
  del(mu2)

