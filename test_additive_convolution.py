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
bin_number = int(50)

for n_example in range(0, 10):
  
  mu1 = sm.spectral_measure()
  mu2 = sm.spectral_measure()
  
  if (n_example == 0):
    mu1.set_semicircle(1)
    mu2.set_semicircle(1)
    N1 = 400
    N2 = 400
    N = 400
    m = 10
    name = "Semicircle+semicircle.eps"
    
  elif (n_example == 1):
    N1 = 400
    N2 = 400
    N = 400
    m = 20
    param = 0.5
    mu1.set_semicircle(1)
    mu2.set_marchenko_pastur(param)
    name = "Semicircle+MP0.5.eps"
    A1 = mu1.get_random_matrix(matrix_size)
    A2 = mu2.get_random_matrix(matrix_size)
    A = A1 + A2
    eigenvalues = scipy.linalg.eigvals(A)
    
  elif (n_example == 2):
    m2 = 1
    mu1.set_semicircle(1)
    mu2.set_uniform(-m2, m2)
    N1 = 400
    N2 = 4000
    N = 400
    m = 20
    print("Theoretical support of the free sum")
    print(1/m2 * numpy.log(m2 + numpy.sqrt(m2**2+1)) + numpy.sqrt(m2**2+1))   
    exact_support = 1/m2 * numpy.log(m2 + numpy.sqrt(m2**2+1)) + numpy.sqrt(m2**2+1)
    name = "Semicircle+Unif1.eps"
    A1 = mu1.get_random_matrix(matrix_size)
    A2 = mu2.get_random_matrix(matrix_size)
    A = A1 + A2
    eigenvalues = scipy.linalg.eigvals(A)
    
  elif (n_example == 3):
    m2 = 4
    mu1.set_semicircle(1)
    mu2.set_uniform(-m2, m2)
    N1 = 400
    N2 = 4000
    N = 400
    m = 20
    print("Theoretical support of the free sum")
    print(1/m2 * numpy.log(m2 + numpy.sqrt(m2**2+1)) + numpy.sqrt(m2**2+1))
    exact_support = 1/m2 * numpy.log(m2 + numpy.sqrt(m2**2+1)) + numpy.sqrt(m2**2+1)
    name = "Semicircle+Unif4.eps"
    A1 = mu1.get_random_matrix(matrix_size)
    A2 = mu2.get_random_matrix(matrix_size)
    A = A1 + A2
    eigenvalues = scipy.linalg.eigvals(A)
    
  elif (n_example == 4):
    mu1.set_uniform(-2, 2)
    mu2.set_marchenko_pastur(0.7)
    N1 = 4000
    N2 = 400
    N = 400
    m = 20
    name = "Unif2+MP0.7.eps"
    Q = numpy.random.normal(0, 1, (matrix_size,matrix_size))
    param = 0.7
    m1 = 2
    Q, R = scipy.linalg.qr(Q)
    d = numpy.random.uniform(-m1, m1, matrix_size)
    A1 = mu1.get_random_matrix(matrix_size)
    A2 = mu2.get_random_matrix(matrix_size)
    A = A1 + A2
    eigenvalues = scipy.linalg.eigvals(A)
    
  elif (n_example == 5):
    mu1.set_semicircle(1)
    mu2.set_weird()
    N1 = 400
    N2 = 4000
    N = 4000
    m = 20
    name = "Semicircle+bad.eps"
    A1 = mu1.get_random_matrix(matrix_size)
    A2 = mu2.get_random_matrix(matrix_size)
    A = A1 + A2
    eigenvalues = scipy.linalg.eigvals(A)

  elif (n_example == 6):
    mu1.set_uniform(-1, 1)
    mu2.set_uniform(-2, 2)
    N1 = 4000
    N2 = 4000
    N = 400
    m = 20
    name = "Unif1+Unif2.eps"
    A1 = mu1.get_random_matrix(matrix_size)
    A2 = mu2.get_random_matrix(matrix_size)
    A = A1 + A2
    eigenvalues = scipy.linalg.eigvals(A)

  elif (n_example == 7):
    mu1.set_semicircle(1)
    m2 = 10
    mu2.set_uniform(-m2, m2)
    N1 = 400
    N2 = 4000
    N = 400
    m = 20
    epsilon = 0.02
    name = "Semicircle+Unif10.eps"
    A1 = mu1.get_random_matrix(matrix_size)
    A2 = mu2.get_random_matrix(matrix_size)
    A = A1 + A2
    eigenvalues = scipy.linalg.eigvals(A)
    print("Theoretical support of the free sum")
    print(1/m2 * numpy.log(m2 + numpy.sqrt(m2**2+1)) + numpy.sqrt(m2**2+1))
    exact_support = 1/m2 * numpy.log(m2 + numpy.sqrt(m2**2+1)) + numpy.sqrt(m2**2+1)
  
  elif (n_example == 8):
    param1 = 0.2
    param2 = 0.6
    epsilon = 0.05
    mu1.set_marchenko_pastur(param1)
    mu2.set_marchenko_pastur(param2)
    N1 = 4000
    N2 = 4000
    N = 4000
    m = 20
    name = "MP+MP.eps"
    A1 = mu1.get_random_matrix(matrix_size)
    A2 = mu2.get_random_matrix(matrix_size)
    A = A1 + A2
    eigenvalues = scipy.linalg.eigvals(A)
    
  elif (n_example == 9):
    mu1.set_semicircle(1)
    weights = numpy.array([1.0, 1.0, 1.0, 1.0])
    weights = weights / sum(weights)
    points = numpy.array([2.0, 4.0, 3.0, 5.0])
    mu2.set_discrete(points, weights)
    N1 = 400
    N2 = 4000
    N = 400
    m = 10
    name = "semicircle+discrete.eps"
    A1 = mu1.get_random_matrix(matrix_size)
    A2 = mu2.get_random_matrix(matrix_size)
    A = A1 + A2
    eigenvalues = scipy.linalg.eigvals(A)
    epsilon = 0.05
  
  print ("-------- Example ", n_example+1, " ----------\n")

  [a_sum, b_sum, t, approx_mu] = freeconvolution.free_additive_convolution(mu1, mu2, N1, N2, N, m, epsilon, do_plots)
  
  if (n_example == 0):
    plt.subplot(4, 3, 10)
    semicircle = sm.spectral_measure()
    semicircle.set_semicircle(numpy.sqrt(2))
    true_mu = semicircle.density(t)
    plt.semilogy(t, abs(approx_mu - true_mu))
  else:
    plt.subplot(4, 3, 10)
    plt.plot(t, approx_mu*matrix_size*(b_sum-a_sum)/bin_number)
    plt.hist(eigenvalues, bins = bin_number)
    
  if (n_example == 2 or n_example == 3 or n_example == 7):
    print(" ------------- ", abs(exact_support + a_sum), abs(exact_support - b_sum), " ---------")
    
  figure = plt.gcf()
  figure.set_size_inches(16, 12)
  plt.savefig(name, dpi=10000)
  # ~ plt.show()
  
  del(mu1)
  del(mu2)

