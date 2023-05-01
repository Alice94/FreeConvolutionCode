# Test 6: Recovering the measure from the (approximate) Cauchy transform

import curves
import spectral_measure as sm
import numpy
import matplotlib.pyplot as plt
import scipy
import freeconvolution

def test_recover_mu_from_G(mu, trueG, n_fourier_coefficients, n_discretization_points, radius):  
  # Draw circle of radius "radius" and compute trueG on the corresponding ellipse
  circle = curves.circle(radius, n_discretization_points)
  ellipse = curves.ellipse(mu.a, mu.b, radius, n_discretization_points)
  image_ellipse = []
  for i in range(round(n_discretization_points)):
    image_ellipse = numpy.append(image_ellipse, trueG(ellipse.discretization[i]))
  
  # do IFFT to get coefficients
  v1 = image_ellipse
  v2 = scipy.fft.ifft(v1)
  N = len(v2)
  v2 = numpy.concatenate((numpy.array([0]), v2[1:n_fourier_coefficients], numpy.zeros(N-n_fourier_coefficients-1)))
  v3 = numpy.array(v2 * numpy.concatenate((numpy.array([0]), 
  numpy.array(numpy.geomspace(radius, (radius)**(n_fourier_coefficients-1), num=n_fourier_coefficients-1)), 
  numpy.zeros(N-n_fourier_coefficients-1))))
  v3 = numpy.append(v3, 0)
  v4 = scipy.fft.fft(v3)
  approx_mu = v4[0:round(N/2)+1]
  # ~ print(approx_mu)
  approx_mu = -numpy.imag(approx_mu)/numpy.pi
  # ~ print(approx_mu)
  t = numpy.linspace(0, numpy.pi, len(approx_mu))
  t = numpy.cos(t) * (mu.b-mu.a)/2 + (mu.b+mu.a)/2
  true_mu = mu.density(t)
  # ~ print(true_mu)
  print("------------------------------------")
  err = abs(true_mu - approx_mu)
  return [t, err, true_mu, approx_mu, v3]

# Number of quadrature points, Fourier coefficients, radius
n_fourier_coefficients = 40
n_quadrature_points = 3000
radius = 1/(0.9)
n_discretization_points = n_quadrature_points

# Semicircle law
mu = sm.spectral_measure()
mu.set_semicircle(1)
trueG = lambda z: (z-numpy.sqrt(z+2) * numpy.sqrt(z-2))/2

plt.subplot(3, 3, 1)
[t, err, true_mu, approx_mu, v3] = test_recover_mu_from_G(mu, trueG, n_fourier_coefficients, n_discretization_points, radius)
plt.plot(t, true_mu)
plt.plot(t, approx_mu)
plt.title('Semicircle')
plt.subplot(3, 3, 2)
plt.semilogy(t, err)
plt.title('Error')
plt.subplot(3, 3, 3)
plt.title('Series coefficients')
plt.plot(v3[1:n_fourier_coefficients], '*')

# Uniform law with m = 3
mu = sm.spectral_measure()
mu.set_uniform(-3, 3)
trueG = lambda z: 1/6 * numpy.log((z+3)/(z-3))

plt.subplot(3, 3, 7)
[t, err, true_mu, approx_mu, v3] = test_recover_mu_from_G(mu, trueG, n_fourier_coefficients, n_discretization_points, radius)
plt.plot(t, true_mu)
plt.plot(t, approx_mu)
plt.title('Uniform on [-3, 3]')
plt.subplot(3, 3, 8)
plt.semilogy(t, err)
plt.subplot(3, 3, 9)
plt.plot(v3[1:n_fourier_coefficients], '*')

# Marchenko-Pastur law
mu = sm.spectral_measure()
param = 0.5
mu.set_marchenko_pastur(param)
trueG = lambda z: (z + param - 1 - numpy.sqrt((z-1-param)-2*numpy.sqrt(param)) * numpy.sqrt((z-1-param)+2*numpy.sqrt(param))) / (2*z*param)

plt.subplot(3, 3, 4)
[t, err, true_mu, approx_mu, v3] = test_recover_mu_from_G(mu, trueG, n_fourier_coefficients, n_discretization_points, radius)
plt.plot(t, true_mu)
plt.plot(t, approx_mu)
plt.title('Marchenko-Pastur lambda=0.5')
plt.subplot(3, 3, 5)
plt.semilogy(t, err)
plt.subplot(3, 3, 6)
plt.plot(v3[1:n_fourier_coefficients], '*')

figure = plt.gcf() # get current figure
figure.set_size_inches(16, 12)
plt.savefig("recover_mu40.eps", dpi=10000)
plt.show()
