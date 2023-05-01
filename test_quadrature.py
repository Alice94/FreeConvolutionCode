# Test 1: mu.compute_G_by_quadrature

import spectral_measure as sm
import numpy
import matplotlib.pyplot as plt
import scipy

# Testing the quadrature rule on measure mu, point z
def test_quadrature(mu, z, trueG, Ns):
  err = []
  print("--------------------------")
  for i in range(T):
    n_quadrature_points = int(Ns[i])
    res = mu.compute_G_by_quadrature(z, n_quadrature_points)
    err = numpy.append(err, abs(res - trueG(z)))
    print(res)
  print("(Theoretically) true result:")
  print(trueG(z))
  return err

# Number of quadrature points, and points z
T = 40
z1 = 1.001
z2 = 0.8 + 0.01*1j

# Semicircle law
Ns = numpy.rint(numpy.linspace(1, 1000, num=T))

mu = sm.spectral_measure()
mu.set_semicircle(1)
trueG = lambda z: (z-numpy.sqrt(z**2-4))/2

plt.subplot(1, 3, 1)
err = test_quadrature(mu, z1*mu.b, trueG, Ns)
plt.semilogy(Ns, err, '-*', label = 'z1')
theory_rate = 1/numpy.abs(mu.joukowski_inverse(numpy.array(z1*mu.b)))

err = test_quadrature(mu, z2*mu.b, trueG, Ns)
plt.semilogy(Ns, err, '-*', label = 'z2')
theory_rate = 1/numpy.abs(mu.joukowski_inverse(numpy.array(z2*mu.b)))
plt.title('Semicircle')
plt.legend()
plt.xlabel('N = # quadr. points')
plt.ylabel('Error')

# Uniform law with m = 3
Ns = numpy.rint(numpy.logspace(1, 6, num=T))

mu = sm.spectral_measure()
mu.set_uniform(-3, 3)
trueG = lambda z: 1/6 * numpy.log((z+3)/(z-3))

plt.subplot(1, 3,3)
err = test_quadrature(mu, z1*mu.b, trueG, Ns)
plt.loglog(Ns, err, '-*', label = 'z1')
theory_rate = 1/numpy.abs(mu.joukowski_inverse(numpy.array(z1*mu.b)))
plt.title('Uniform on [-3, 3]')

err = test_quadrature(mu, z2*mu.b, trueG, Ns)
plt.loglog(Ns, err, '-*', label = 'z2')
theory_rate = 1/numpy.abs(mu.joukowski_inverse(numpy.array(z2*mu.b)))
plt.legend()
plt.xlabel('N = # quadr. points')
plt.ylabel('Error')


# Marchenko-Pastur law
Ns = numpy.rint(numpy.linspace(1, 1000, num=T))

mu = sm.spectral_measure()
param = 0.8
mu.set_marchenko_pastur(param)
trueG = lambda z: (z + param - 1 - numpy.sqrt((z-1-param)**2-4*param)) / (2*z*param)

t = numpy.array(numpy.linspace(mu.a, mu.b, 1000))
d = mu.density(t)
print("Check that it is a probability measure")
print(sum(d) * (mu.b-mu.a)/1000)

plt.subplot(1, 3, 2)
err = test_quadrature(mu, z1*mu.b, trueG, Ns)
plt.semilogy(Ns, err, '-*', label = 'z1')
theory_rate = 1/numpy.abs(mu.joukowski_inverse(numpy.array(z1*mu.b)))
plt.title('Marchenko-Pastur lambda=0.8')

err = test_quadrature(mu, z2*mu.b, trueG, Ns)
plt.semilogy(Ns, err, '-*', label = 'z2')
theory_rate = 1/numpy.abs(mu.joukowski_inverse(numpy.array(z2*mu.b)))
plt.legend()
plt.xlabel('N = # quadr. points')
plt.ylabel('Error')

figure = plt.gcf() 
figure.set_size_inches(16, 5)
plt.savefig("quadrature.eps", dpi=10000)
plt.show()
