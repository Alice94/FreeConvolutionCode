import numpy
import matplotlib.pyplot as plt
import scipy
from random import choices

class spectral_measure:
	
  ##### Basic functions #####
  
  def set_support(self, a, b):
    self.a = a
    self.b = b
    
  def set_density(self, density):
    self.density = density
  
  def set_atoms(self, atoms, weights):
    self.atoms = atoms
    self.weights = weights
  
  ##### Special measures #####
  
  def set_semicircle(self, k):
    # Semicircle with parameter k (the standard one is with k=1)
    self.a = -2*k
    self.b = 2*k
    self.k = k
    self.density = lambda t: numpy.nan_to_num(numpy.sqrt(4*(k**2) - t**2)/(2*numpy.pi)/k**2) * (4*(k**2) - t**2 > 1e-10)
    self.name = "semicircle"
    
  def set_shifted_semicircle(self, k):
    # Shifted semicircle distribution (all eigenvalues are positive and bounded away from zero)
    self.a = k
    self.b = 5*k
    self.k = k
    self.density = lambda t: numpy.nan_to_num(numpy.sqrt(4*(k**2) - (t-3*k)**2)/(2*numpy.pi)/k**2) * (4*(k**2) - (t-3*k)**2 > 1e-10)
    self.name = "shifted_semicircle"
    
  def set_truncated_semicircle(self, k):
    # Semicircle distribution in which we truncate the negative part and we accumulated it in zero
    self.a = 0
    self.b = 2*k
    self.k = k
    self.density = lambda t: numpy.nan_to_num(numpy.sqrt(4*(k**2) - t**2)/(2*numpy.pi)/k**2) * (4*(k**2) - t**2 > 1e-10)
    self.atoms = numpy.array([0])
    self.weights = numpy.array([0.5])
    self.name = "truncated_semicircle"
    
  def set_truncated_semicircle_2(self, k):
    # Shifted semicircle distribution in which we truncate to 3k all eigenvalues smaller than 3k
    self.a = 3*k
    self.b = 5*k
    self.k = k
    self.density = lambda t: numpy.nan_to_num(numpy.sqrt(4*(k**2) - (t-3*k)**2)/(2*numpy.pi)/k**2) * (4*(k**2) - (t-3*k)**2 > 1e-10)
    self.atoms = numpy.array([3*k])
    self.weights = numpy.array([0.5])
    self.name = "truncated_semicircle_2"
  
  def set_uniform(self, a, b):
    # Uniform distribution on the interval [a, b]
    self.a = a
    self.b = b
    self.density = lambda t: 1/(b-a) * numpy.ones(len(numpy.atleast_1d(t)))
    self.name = "uniform"
  
  def set_marchenko_pastur(self, param):
    # Marchenko-Pastur sitribution with paramter "param" in (0,1)
    paramplus = (1 + numpy.sqrt(param))**2
    paramminus = (1 - numpy.sqrt(param))**2
    self.a = paramminus
    self.b = paramplus
    self.param = param
    self.density = lambda t: numpy.nan_to_num(1/(2*numpy.pi * t * param) * numpy.sqrt((paramplus - t) *
    (t - paramminus))) * (paramplus - t > 1e-10) * (t - paramminus > 1e-10) * (abs(t) > 1e-10)
    self.name = "marchenko_pastur"
    
  def set_weird(self):
    # Measure that is fun to play with (Cauchy transform is not invertible)
    self.a = -numpy.sqrt(3)
    self.b = numpy.sqrt(3)
    self.density = lambda t: 5 * numpy.sqrt(3) / (24*6) * (t**2 + 1)**2
    self.name = "weird"
    
  def set_discrete(self, atoms, weights):
    # Discrete measure for which we prescribe some weights in some points ("atoms"). sum(weights) must be 1
    self.a = min(atoms)
    self.b = max(atoms)
    self.atoms = atoms
    self.weights = weights
    self.name = "discrete"

  ##### Transforms (Cauchy, R, S, T, and their inverses) #####
  
  # Setting up the computation of the Cauchy transform (and its derivative) via a discretization of the integral with trapezoidal quadrature rule
  # Converges exponentially fast in the number of quadrature points if the measure has sqrt-decay at the boundary
  # The (possible) atoms are considered separately (Cauchy transform is explicitly known for the atomic part)  
  def set_G(self, N):
    if (hasattr(self, "density")):
      t1 = numpy.array(numpy.linspace(0, 1, N+1))
      t2 = numpy.cos(t1 * numpy.pi) * (self.b - self.a)/2 + (self.b + self.a)/2
      t3 = numpy.nan_to_num(self.density(t2) * numpy.array(numpy.sin(numpy.pi * t1)) * (self.b-self.a)/2 * numpy.pi)/N
      if (hasattr(self, "atoms")):
        self.G = lambda z: sum(t3 / (z - t2)) - t3[0]/(2*(z-t2[0])) - t3[N]/(2*(z - t2[N])) + sum(self.weights / (z - self.atoms))
        self.G_prime = lambda z: -sum(t3 / (z - t2)**2) + t3[0]/(2*(z-t2[0])**2) + t3[N]/(2*(z - t2[N])**2) - sum(self.weights / (z - self.atoms)**2)
      else:
        self.G = lambda z: sum(t3 / (z - t2)) - t3[0]/(2*(z-t2[0])) - t3[N]/(2*(z - t2[N]))
        self.G_prime = lambda z: -sum(t3 / (z - t2)**2) + t3[0]/(2*(z-t2[0])**2) + t3[N]/(2*(z - t2[N])**2) 
    else:
      self.G = lambda z: sum(self.weights / (z - self.atoms))
      self.G_prime = lambda z: - sum(self.weights / (z - self.atoms)**2)
  
  # Setting up the computation of the T-transform (and its derivative) via a discretization of the integral with trapezoidal quadrature rule
  # Converges exponentially fast in N (number of quadrature points) if the measure has sqrt-decay at the boundary
  # The (possible) atoms are considered separately (T-transform is explicitly known for the atomic part)      
  def set_T(self, N):
    if (hasattr(self, "density")):
      t1 = numpy.array(numpy.linspace(0, 1, N+1))
      t2 = numpy.cos(t1 * numpy.pi) * (self.b - self.a)/2 + (self.b + self.a)/2
      t3 = numpy.nan_to_num(self.density(t2) * numpy.array(numpy.sin(numpy.pi * t1)) * (self.b-self.a)/2 * numpy.pi * t2)/N
      if (hasattr(self, "atoms")):
        self.T = lambda z: sum(t3 / (z - t2)) - t3[0]/(2*(z-t2[0])) - t3[N]/(2*(z - t2[N])) + sum(self.weights * self.atoms / (z - self.atoms))
        self.T_prime = lambda z: -sum(t3 / (z - t2)**2) + t3[0]/(2*(z-t2[0])**2) + t3[N]/(2*(z - t2[N])**2) - sum(self.weights * self.atoms / (z - self.atoms)**2)
      else:
        self.T = lambda z: sum(t3 / (z - t2)) - t3[0]/(2*(z-t2[0])) - t3[N]/(2*(z - t2[N]))
        self.T_prime = lambda z: -sum(t3 / (z - t2)**2) + t3[0]/(2*(z-t2[0])**2) + t3[N]/(2*(z - t2[N])**2)
    else:
      self.T = lambda z: sum(self.weights * self.atoms / (z - self.atoms))
      self.T_prime = lambda z: - sum(self.weights * self.atoms / (z - self.atoms)**2)
  
  # Computation of Cauchy transform in a single point, via trapezoidal quadrature rule (mainly for debugging)
  def compute_G_by_quadrature(self, z, n_quadrature_points):
    if (hasattr(self, "density")):
      t1 = numpy.array(numpy.linspace(0, 1, n_quadrature_points+1))
      t2 = numpy.cos(t1 * numpy.pi) * (self.b - self.a)/2 + (self.b + self.a)/2
      t3 = numpy.array(numpy.sin(numpy.pi * t1))
      u = numpy.nan_to_num(self.density(t2) / (z - t2) * t3 * (self.b-self.a)/2 * numpy.pi) * (abs(z-t2) > 1e-10)  
      if (hasattr(self, "atoms")):   
        return (sum(u) - u[0]/2 - u[n_quadrature_points]/2)/n_quadrature_points + sum(self.weights / (z - self.atoms))
      else:
        return (sum(u) - u[0]/2 - u[n_quadrature_points]/2)/n_quadrature_points
    else:
      return sum(self.weights / (z - self.atoms))
  
  # Computation of T-transform in a single point, via trapezoidal quadrature rule (mainly for debugging)  
  def compute_T_by_quadrature(self, z, n_quadrature_points):
    if (hasattr(self, "density")):
      t1 = numpy.array(numpy.linspace(0, 1, n_quadrature_points+1))
      t2 = numpy.cos(t1 * numpy.pi) * (self.b - self.a)/2 + (self.b + self.a)/2
      t3 = numpy.array(numpy.sin(numpy.pi * t1))
      u = numpy.nan_to_num(self.density(t2) / (z - t2) * t3 * (self.b-self.a)/2 * numpy.pi * t2) * (abs(z-t2) > 1e-10)
      if (hasattr(self, "atoms")):   
        return (sum(u) - u[0]/2 - u[n_quadrature_points]/2)/n_quadrature_points + sum(self.weights * self.atoms / (z - self.atoms))
      else:
        return (sum(u) - u[0]/2 - u[n_quadrature_points]/2)/n_quadrature_points
    else:
      return sum(self.weights * self.atoms / (z - self.atoms))
    
  # The inverse of G is computed by numerical quadrature applied to the Cauchy integral theorem applied to
  # Gtilde(circle of radius r) = G(J(circle of radius r))
  # This function computes the coefficients appearing in the trapezoidal quadrature formula
  # N = number of quadrature points (e.g. 400 if very regular, 4000 otherwise)
  # r = radius of the circle that defines the curve on which we apply the Cauchy integral theorem (e.g. 0.95)
  def set_G_inverse(self, N, r):
    v1 = numpy.array(r * numpy.exp(2*numpy.pi * 1j * numpy.double(range(0, N)) / N)) # points on the circle
    v2 = self.joukowski(v1)
    v3 = []
    v4 = []
    for i in range(N):
      v3 = numpy.append(v3, self.G_tilde(v1[i]))
      v4 = numpy.append(v4, self.G_tilde_prime(v1[i]))
    v3 = numpy.array(v3)
    v4 = numpy.array(v4)
    self.image_circle = numpy.append(v3, v3[0])
    self.G_inverse = lambda z: 1/z + 1/N * sum(v1 * v4 * (v2 - 1/v3) / (v3 - z))
  
  # The inverse of T is computed by numerical quadrature applied to the Cauchy integral theorem applied to
  # Ttilde(circle of radius r) = T(J(circle of radius r))
  # This function computes the coefficients appearing in the trapezoidal quadrature formula
  # N = number of quadrature points (e.g. 400 if very regular, 4000 otherwise)
  # r = radius of the circle that defines the curve on which we apply the Cauchy integral theorem (e.g. 0.95)  
  def set_T_inverse(self, N, r):
    v1 = numpy.array(r * numpy.exp(2*numpy.pi * 1j * numpy.double(range(0, N)) / N)) # points on the circle
    v2 = self.joukowski(v1)
    v3 = []
    v4 = []
    for i in range(N):
      v3 = numpy.append(v3, self.T_tilde(v1[i]))
      v4 = numpy.append(v4, self.T_tilde_prime(v1[i]))
    v3 = numpy.array(v3)
    v4 = numpy.array(v4)
    self.image_circle = numpy.append(v3, v3[0])
    self.S = lambda z: - 1/N * sum(v1 * v4 * (1+v3) / (v2 * v3 * (z - v3)))
    self.T_inverse = lambda z: (1 + z) / (z * self.S(z))
  
  # The Joukowski transform (relative to the segment [a,b] = support of mu)
  # maps the unit disk to C U {\infty} \ [a,b] (conformal map).
  def joukowski(self, z):
    return 1/2 * (z + 1/z) * (self.b - self.a)/2 + (self.b + self.a)/2
  
  def joukowski_prime(self, z):
    return 1/2 * (self.b - self.a)/2 * (1 - 1/z**2)
  
  def joukowski_inverse(self, y): # returns the value inside the unit disk
    alpha = (self.b-self.a)/2
    beta = (self.b+self.a)/2
    res = (y-beta)/alpha + numpy.sqrt(((y.astype(complex)-beta)/alpha)**2-1)
    if numpy.isscalar(res):
      if (abs(res) > 1+1e-10 or (abs(abs(res)-1) < 1e-10 and numpy.imag(res) < 0)):
        return 1/res
      else:
        return res
    for j in range(len(res)):
      if (abs(res[j]) > 1 + 1e-10 or (abs(abs(res[j])-1) < 1e-10 and numpy.imag(res[j]) < 0)):
        res[j] = 1/res[j]
    return res
      
  def joukowski_inverse_prime(self, y):
    return 1/self.joukowski_prime(self.joukowski_inverse(y))
  
  # G_tilde is the composition of G and J and it's analytic from the unit disk to C
  def G_tilde(self, z):
    return self.G(self.joukowski(z))
    
  def G_tilde_prime(self, z):
    return self.G_prime(self.joukowski(z)) * self.joukowski_prime(z)
    
  def G_inverse_prime(self, z):
    return 1/self.G_prime(self.G_inverse(z))
  
  # T_tilde is the composition of G and J and it's analytic from the unit disk to C  
  def T_tilde(self, z):
    return self.T(self.joukowski(z))
    
  def T_tilde_prime(self, z):
    return self.T_prime(self.joukowski(z)) * self.joukowski_prime(z)
    
  def T_inverse_prime(self, z):
    return 1/self.T_prime(self.T_inverse(z))
    
  ##### Plotting the density #####
  
  def plot(self, col, lab):
    # col = color (e.g. "b")
    # lab = label (e.g. "mu1")
    t = numpy.array(numpy.linspace(self.a, self.b, 1000))
    t = numpy.linspace(0, numpy.pi, 53)
    t = numpy.cos(t) * (self.b-self.a)/2 + (self.a+self.b)/2
    if (hasattr(self, "density")):
      d = list(map(self.density, t))
      if (hasattr(d, "__len__") == 0):
        d = numpy.ones(len(t)) * d
      plt.plot(t, d, col, label=lab)
    if (hasattr(self, "atoms")):
      plt.plot(self.atoms, numpy.zeros(len(self.atoms)), '*', label = lab)
      
  ##### Building random matrices #####
  
  # Given a matrix size, construct a random matrix that, asymptotically, has eigenvalue distribution mu
  
  def get_random_matrix(self, matrix_size):
    if (self.name == "semicircle"):
      A = numpy.random.normal(0, 1, (matrix_size, matrix_size))
      A = numpy.triu(A)/numpy.sqrt(matrix_size)
      A = self.k * (A + A.transpose())
      
    elif (self.name == "shifted_semicircle"):
      A = numpy.random.normal(0, 1, (matrix_size, matrix_size))
      A = numpy.triu(A)/numpy.sqrt(matrix_size)
      A = self.k * (A + A.transpose() + 3*numpy.identity(matrix_size))
      
    elif (self.name == "truncated_semicircle"):
      A = numpy.random.normal(0, 1, (matrix_size, matrix_size))
      A = numpy.triu(A)/numpy.sqrt(matrix_size)
      A = A + A.transpose()
      W, V = scipy.linalg.eig(A)
      W = W * (W > 0)
      A = numpy.matmul(V, numpy.diag(W))
      A = numpy.matmul(A, V.transpose())
      A = self.k * A
      
    elif (self.name == "truncated_semicircle_2"):
      A = numpy.random.normal(0, 1, (matrix_size, matrix_size))
      A = numpy.triu(A)/numpy.sqrt(matrix_size)
      A = A + A.transpose()
      W, V = scipy.linalg.eig(A)
      W = W * (W > 0)
      A = numpy.matmul(V, numpy.diag(W))
      A = numpy.matmul(A, V.transpose())
      A = self.k * (A + 3 * numpy.identity(matrix_size))
      
    elif (self.name == "uniform"):
      Q = numpy.random.normal(0, 1, (matrix_size,matrix_size))
      Q, R = scipy.linalg.qr(Q)
      d = numpy.random.uniform(self.a, self.b, matrix_size)
      A = numpy.matmul(Q, numpy.diag(d))
      A = numpy.matmul(A, Q.transpose())
      
    elif (self.name == "marchenko_pastur"):
      A = numpy.random.normal(0, 1, (matrix_size, int(numpy.round(matrix_size/self.param))))
      A = A/numpy.sqrt(matrix_size/self.param)
      A = numpy.matmul(A, A.transpose())
      
    elif (self.name == "weird"):
      population = numpy.linspace(self.a, self.b, 10000)
      weights = self.density(population)
      weights = weights / numpy.sum(weights)
      Q = numpy.random.normal(0, 1, (matrix_size,matrix_size))
      Q, R = scipy.linalg.qr(Q)
      d = numpy.random.uniform(-1, 1, matrix_size)
      for i in range(matrix_size):
        xx = choices(population, weights)
        d[i] = xx[0]
      A = numpy.matmul(Q, numpy.diag(d))
      A = numpy.matmul(A, Q.transpose())
      
    elif (self.name == "discrete"):
      Q = numpy.random.normal(0, 1, (matrix_size,matrix_size))
      Q, R = scipy.linalg.qr(Q)
      d = numpy.random.uniform(-1, 1, matrix_size)
      for i in range(matrix_size):
        xx = choices(self.atoms, self.weights)
        d[i] = xx[0]
      A = numpy.matmul(Q, numpy.diag(d))
      A = numpy.matmul(A, Q.transpose())
      
    else:
      print("Error: generation of random matrix is not supported for this distribution")
      A = numpy.random.normal(0, 1, (matrix_size, matrix_size))
    
    return A
