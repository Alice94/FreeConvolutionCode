import curves
import spectral_measure as sm
import numpy
import matplotlib.pyplot as plt
import scipy

def derivative_G_inverse_sum(y, mu1, mu2):
  x1 = mu1.G_inverse_prime(y)
  x2 = mu2.G_inverse_prime(y)
  return (1/y**2 + x1 + x2)

# Find zero derivative of G_mu^{-1} in an interval
def find_zero_derivative_g(left, right, mu1, mu2, tol):
  der_left = numpy.real(derivative_G_inverse_sum(left, mu1, mu2))
  der_right = numpy.real(derivative_G_inverse_sum(right, mu1, mu2))
  if (left > right):
    print("---- Error: left > right")
  if (der_left * der_right > 0):
    print(" ##### Problem: there might be no zero derivative in the selected interval! Returning one of the extrema...")
    if (abs(der_left) > abs(der_right)):
      return right
    else:
      return left
  medium = (right+left)/2
  while (right - left > tol):
    medium = (right+left)/2
    der_medium = numpy.real(derivative_G_inverse_sum(medium, mu1, mu2))
    if (der_medium * der_left < 0):
      right = medium
    else:
      left = medium
  return medium

def derivative_T_inverse_prod(y, mu1, mu2):
  x1 = mu1.T_inverse(y)
  x2 = mu2.T_inverse(y)
  z1 = mu1.T_inverse_prime(y)
  z2 = mu2.T_inverse_prime(y)
  res = ((x1*z2 + x2*z1) * y / (1+y) + x1*x2 / (1+y)**2)
  return res
  
# Find zero derivative of T_mu^{-1} in an interval
def find_zero_derivative_t(left, right, mu1, mu2, tol):
  der_left = numpy.real(derivative_T_inverse_prod(left, mu1, mu2))
  der_right = numpy.real(derivative_T_inverse_prod(right, mu1, mu2))
  
  if (left > right):
    print("---- Error: left > right")
  if (der_left * der_right > 0):
    print(" ##### Problem: there might be no zero derivative in the selected interval! Returning one of the extrema...")
    if (abs(der_left) > abs(der_right)):
      return right
    else:
      return left
  
  medium = (right+left)/2
  while (right - left > tol):
    medium = (right+left)/2
    der_medium = numpy.real(derivative_T_inverse_prod(medium, mu1, mu2))
    if (der_medium * der_left < 0):
      right = medium
    else:
      left = medium
  return medium

# Compute the free additive convolution of two measures
def free_additive_convolution(mu1, mu2, N1, N2, N, m, epsilon, do_plots):  
  # mu1, mu2 = input measures
  # N1 = quadrature points for G1, G1prime, G1inverse
  # N2 = quadrature points for G2, G2prime, G2inverse
  # N = quadrature points for Cauchy integral that defines G
  # m = number of series coefficients that we save (Ideally this should depend on r_C)
  # epsilon = how close to "the border" do we want to get. Ideally between 0.01 and 0.1
  # if do_plots == 1 it draws plots, otherwise not
  
  # STEP 1: Set up the Cauchy transform and its inverse for mu1 and mu2 (if not already known)
  print("STEP 1: Setting up the R-transform for mu1 and mu2")
  r_A = 1-epsilon 
  if ((not hasattr(mu1, "G")) or (not hasattr(mu1, "G_prime"))):
    print("-- Setting up G and G_prime for mu1")
    mu1.set_G(N1)
  if (not hasattr(mu1, "G_inverse")):
    print("-- Setting up G_inverse for mu1")
    mu1.set_G_inverse(N, r_A) # This was N1
    Gamma1 = mu1.image_circle
    Gamma1 = curves.curve(len(Gamma1)-1, Gamma1)
  else:
    Gamma1 = curves.circle(r_A, N) # This was N1
    for i in range(N+1): # This was N1
      Gamma1.discretization[i] = mu1.G_tilde(Gamma1.discretization[i])
      
  if ((not hasattr(mu2, "G")) or (not hasattr(mu2, "G_prime"))):
    print("-- Setting up G and G_prime for mu2")
    mu2.set_G(N2)
  if (not hasattr(mu2, "G_inverse")):
    print("-- Setting up G_inverse for mu2")
    mu2.set_G_inverse(N, r_A)
    Gamma2 = mu2.image_circle
    Gamma2 = curves.curve(len(Gamma2)-1, Gamma2)
  else:
    Gamma2 = curves.circle(r_A, N)
    for i in range(N+1):
      Gamma2.discretization[i] = mu2.G_tilde(Gamma2.discretization[i])
      
  # STEP 2: Computing the support of mu
  print("STEP 2: Computing support of mu")
  # First part: Compute zeros of the derivative of G_mu inverse
  a1 = mu1.G_tilde(-r_A + epsilon)
  b1 = mu1.G_tilde(r_A - epsilon)
  a2 = mu2.G_tilde(-r_A + epsilon)
  b2 = mu2.G_tilde(r_A - epsilon)
  
  a0 = max(numpy.real(a1), numpy.real(a2))
  b0 = min(numpy.real(b1), numpy.real(b2))
  tol = 10**(-12) 
  
  a_sum_cd = find_zero_derivative_g(a0, a0*epsilon, mu1, mu2, tol)
  b_sum_cd = find_zero_derivative_g(b0*epsilon, b0, mu1, mu2, tol)  
  
  # Second part: Actually compute support of mu
  x1 = mu1.G_inverse(a_sum_cd)
  y1 = mu1.G_inverse(b_sum_cd)
  x2 = mu2.G_inverse(a_sum_cd)
  y2 = mu2.G_inverse(b_sum_cd)
  a_sum = numpy.real(-1/a_sum_cd + x1 + x2)
  b_sum = numpy.real(-1/b_sum_cd + y1 + y2)
  print("-- Computed support of mu: [", a_sum, ", ", b_sum, "]")
  muplus = sm.spectral_measure()
  muplus.set_support(a_sum, b_sum)
  
  # STEP 3: Evaluating G_mu on a suitable curve
  print("STEP 3: Evaluating G_mu on a suitable curve")
  
  # Compute small circle inside and G^{-1}(circle)
  print("-- Computing orange circle")
  # Compute the largest circle that fits inside Gamma1 and Gamma2
  r_B = min(min(abs(numpy.array(Gamma1.discretization))), min(abs(numpy.array(Gamma2.discretization))))
  r_B = min(r_B, min(abs(a_sum_cd), abs(b_sum_cd)))
  r_B = (1-epsilon) * r_B
  Csum = curves.circle(1, N)
  # Shrink the circle in order to get it inside the region where G is invertible
  for i in range(Csum.n_points+1):
    A = Csum.discretization[i] * r_B
    if (abs(numpy.imag(A)) < 0.00001):
      continue
    a1 = mu1.G_inverse(A)
    a2 = mu2.G_inverse(A)
    a_sum2 = a1 + a2 - 1/A
    if (numpy.imag(a_sum2) * numpy.imag(A) > 0): 
      # it means that the largest point does not work and we need to restrict the interval
      # Binary search
      tol = 0.001
      C = (1-epsilon)*A # here I am assuming that C is inside the analiticity region
      while (abs(A-C) > tol):
        B = (A+C)/2
        b1 = mu1.G_inverse(B)
        b2 = mu2.G_inverse(B)
        b_sum2 = b1 + b2 - 1/B
        if (numpy.imag(b_sum2) * numpy.imag(B) > 0):
          A = B
        else:
          C = B
      r_B = abs(B)
  
  # Now resize the circle, such that it is inside the analiticity region
  Csum = curves.circle(r_B, N)
  
  # Compute preimage of Csum with the sum  
  G1invCsum = []
  G2invCsum = []
  for i in range(N+1):
    G1invCsum.append(mu1.G_inverse(Csum.discretization[i]))
    G2invCsum.append(mu2.G_inverse(Csum.discretization[i]))
  G1invCsum = curves.curve(N, numpy.array(G1invCsum))
  G2invCsum = curves.curve(N, numpy.array(G2invCsum))
  
  # Sum  
  res = G1invCsum.discretization + G2invCsum.discretization - 1/Csum.discretization
  GinvCsum = curves.curve(N, res)
  
  # Choose ellipse outside the small circle Csum and extract info on mu from it
  print("-- Choosing green circle A")
  # First of all, choose an elliptical contour OUTSIDE the GinvCsum
  # Compute maximum "radius" corresponding to GinvCsum = minimum radius of JinvGinvCsum
  JinvGinvCsum = curves.curve(N, numpy.zeros(N + 1, dtype=complex))
  for i in range(N+1):
    JinvGinvCsum.discretization[i] = muplus.joukowski_inverse(GinvCsum.discretization[i])
  r_C = min(abs(numpy.array(JinvGinvCsum.discretization))) * (1 - epsilon) 
  
  # Choose a suitable number of points for the discretization of the green circle
  M = max(100, 2 * round(8 * numpy.log(10) / numpy.log(1/r_C)))
  m = min(m, round(14 * numpy.log(10) / numpy.log(1/r_C)))
  print(" --Updated values: M = ", M, ", m = ", m)  
  JA = curves.ellipse(a_sum, b_sum, r_C, M)
  
  # Evaluate G_mu on the JA using Cauchy integral,
  # knowing that GinvCsum is sent by G_mu into Csum
  # (remember that Csum has radius small_radius)
  print("-- Evaluating G on green ellipse")
  w1 = Csum.discretization[range(0, N)]
  w2 = GinvCsum.discretization[range(0, N)]
  w3 = []
  for i in range(0, N):
    w3 = numpy.append(w3, derivative_G_inverse_sum(w1[i], mu1, mu2))
  GA = []
  
  for i in range(0, M):
    der = - sum((w1**2) * w3 / (w2 - JA.discretization[i]))/N
    GA = numpy.append(GA, der)
  GA = numpy.append(GA, GA[0])
  GA = curves.curve(M,GA)
  
  A = curves.circle(r_C, M)
  
  # Get discretization of measure mu  
  unitDisk = curves.circle(1, M)
  
  # STEP 4: Recover density of mu from its Cauchy transform
  print("STEP 4: Recover density of mu from its Cauchy transform")
  
  # Get discretization of measure mu and plot it
  v1 = GA.discretization[1 : M+1]
  v1 = numpy.flip(v1)
  v2 = scipy.fft.ifft(v1)
  v2 = numpy.concatenate((numpy.array([0]), v2[1:m], numpy.zeros(len(v2)-m)))
  v3 = numpy.array(v2 * numpy.concatenate((numpy.array([0]), 
  numpy.array(numpy.geomspace(1/r_C, (1/r_C)**(m-1), num=m-1)), 
  numpy.zeros(len(v2)-m))))
  
  g_coefficients = numpy.array(v3[1:m])
  g_coefficients = numpy.transpose(g_coefficients)
  density_sum = numpy.vectorize(lambda t: -1/numpy.pi * numpy.imag(g_coefficients @ ((muplus.joukowski_inverse(t))**range(1,m)))) 
  muplus.set_density(density_sum)
  
  v4 = scipy.fft.fft(v3)
  mu = v4[0:round(len(v2)/2)+1]
  
  alpha = (b_sum - a_sum)/2
  beta = (b_sum + a_sum)/2
  
  t = numpy.linspace(0, numpy.pi, len(mu))
  t = numpy.cos(t) * alpha + beta
  approx_mu = numpy.imag(mu)/numpy.pi
  
  # Do all the plots
  if (do_plots == 1):
    plt.figure()
    for i in [1,2,3,4,5,6,7,8,9,12]:
      plt.subplot(4, 3, i)
      plt.axis('equal')
      plt.grid(True, which='both')
      plt.axhline(y=0, color='k')
      plt.axvline(x=0, color='k')
    
    plt.subplot(4, 3, 2)
    J1C = curves.ellipse(mu1.a, mu1.b, r_A, N1)
    J1C.plot('blue')
    G1invCsum.plot('orange')
    
    plt.subplot(4, 3, 1)
    J1invG1invCsum = G1invCsum
    for i in range(J1invG1invCsum.n_points+1):
      J1invG1invCsum.discretization[i] = mu1.joukowski_inverse(J1invG1invCsum.discretization[i])
    J1invG1invCsum.plot('orange')
    unitCircle = curves.circle(1, N)
    unitCircle.plot('red')
    C = curves.circle(1-epsilon, N)
    C.plot('blue')
    
    plt.subplot(4, 3, 3)
    Gamma1.plot('blue')
    Csum.plot('orange')
    
    plt.subplot(4, 3, 5)
    J2C = curves.ellipse(mu2.a, mu2.b, r_A, N2)
    J2C.plot('blue')
    G2invCsum.plot('orange')
    
    plt.subplot(4, 3, 4)
    J2invG2invCsum = G2invCsum
    for i in range(J2invG2invCsum.n_points+1):
      J2invG2invCsum.discretization[i] = mu2.joukowski_inverse(J2invG2invCsum.discretization[i])
    J2invG2invCsum.plot('orange')
    unitCircle.plot('red')
    C.plot('blue')
    
    plt.subplot(4, 3, 6)
    Gamma2.plot('blue')
    Csum.plot('orange')
    
    plt.subplot(4, 3, 8)
    plt.plot(a_sum, 0, 'ro')
    plt.plot(numpy.real(b_sum), 0, 'ro')
    GinvCsum.plot('orange')
    JA.plot('green')
    
    plt.subplot(4, 3, 7)
    unitCircle.plot('red')
    # Plot A
    A.plot('green')
    # Plot Gamma
    JinvGinvCsum.plot('orange')
    
    plt.subplot(4, 3, 9)
    plt.plot(numpy.real(a_sum_cd), 0, 'ro')
    plt.plot(numpy.real(b_sum_cd), 0, 'ro')
    GA.plot('green')
    Csum.plot('orange')
    
    plt.subplot(4, 3, 11)
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.plot(t, approx_mu, 'red', label = 'free sum')
    mu1.plot('blue', 'mu_1')
    mu2.plot('green', 'mu_2')
    plt.legend()
    
    plt.subplot(4, 3, 12)
    Gamma1.plot('blue')
    Gamma2.plot('blue')
    Csum.plot('orange')
    
  if (do_plots == 2): # plot things for the mu1 only
    plt.figure()
    for i in [1,2,3]:
      plt.subplot(1, 3, i)
      plt.axis('equal')
      plt.grid(True, which='both')
      plt.axhline(y=0, color='k')
      plt.axvline(x=0, color='k')
      
    plt.subplot(1, 3, 2)
    J1C = curves.ellipse(mu1.a, mu1.b, r_A, N1)
    J1C.plot('blue')
    
    plt.subplot(1, 3, 1)
    unitCircle = curves.circle(1, N)
    unitCircle.plot('red')
    C = curves.circle(1-epsilon, N)
    C.plot('blue')
    
    plt.subplot(1, 3, 3)
    Gamma1.plot('blue')
    
  if (do_plots == 3):
    plt.figure()
    for i in [1,2,3]:
      plt.subplot(1, 3, i)
      plt.axis('equal')
      plt.grid(True, which='both')
      plt.axhline(y=0, color='k')
      plt.axvline(x=0, color='k')
    
    unitCircle = curves.circle(1, N)  
    plt.subplot(1, 3, 2)
    plt.plot(a_sum, 0, 'ro')
    plt.plot(numpy.real(b_sum), 0, 'ro')
    GinvCsum.plot('orange')
    JA.plot('green')
    
    plt.subplot(1, 3, 1)
    unitCircle.plot('red')
    # Plot A
    A.plot('green')
    # Plot Gamma
    JinvGinvCsum.plot('orange')
    
    plt.subplot(1, 3, 3)
    plt.plot(numpy.real(a_sum_cd), 0, 'ro')
    plt.plot(numpy.real(b_sum_cd), 0, 'ro')
    GA.plot('green')
    Csum.plot('orange')
    
  if do_plots == 4:
    plt.subplot(1, 2, 2)
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.plot(t, approx_mu, 'red', label = 'free sum')
    mu1.plot('blue', 'mu_1')
    mu2.plot('green', 'mu_2')
    plt.legend()

  return [a_sum, b_sum, t, approx_mu, muplus]
  
# Compute the free multiplicative convolution of two measures
def free_multiplicative_convolution(mu1, mu2, N1, N2, N, m, epsilon, do_plots):  
  if do_plots > 0:
    plt.figure()
  # mu1, mu2 = input measures
  # N1 = quadrature points for G1, G1prime, G1inverse
  # N2 = quadrature points for G2, G2prime, G2inverse
  # N = quadrature points for Cauchy integral that defines G
  # m = number of series coefficients that we save (Ideally this should depend on r_C)
  # epsilon = how close to "the border" do we want to get. Ideally between 0.01 and 0.1
  # if do_plots == 1 it draws plots, otherwise not
  
  # STEP 1: Set up the T-transform and its inverse for mu1 and mu2 (if not already known)
  print("STEP 1: Setting up the S-transform for mu1 and mu2")
  r_A = 1-epsilon 
  if ((not hasattr(mu1, "T")) or (not hasattr(mu1, "T_prime"))):
    print("-- Setting up T and T_prime for mu1")
    mu1.set_T(N1)
  if (not hasattr(mu1, "T_inverse")):
    print("-- Setting up T_inverse for mu1")
    mu1.set_T_inverse(N1, r_A)
    Gamma1 = mu1.image_circle
    Gamma1 = curves.curve(len(Gamma1)-1, Gamma1)
  else:
    Gamma1 = curves.circle(r_A, N1)
    for i in range(N1+1):
      Gamma1.discretization[i] = mu1.T_tilde(Gamma1.discretization[i])
      
  if ((not hasattr(mu2, "T")) or (not hasattr(mu2, "T_prime"))):
    print("-- Setting up T and T_prime for mu2")
    mu2.set_T(N2)
  if (not hasattr(mu2, "T_inverse")):
    print("-- Setting up T_inverse for mu2")
    mu2.set_T_inverse(N2, r_A)
    Gamma2 = mu2.image_circle
    Gamma2 = curves.curve(len(Gamma2)-1, Gamma2)
  else:
    Gamma2 = curves.circle(r_A, N2)
    for i in range(N2+1):
      Gamma2.discretization[i] = mu2.T_tilde(Gamma2.discretization[i])
      
  # STEP 2: Computing the support of mu
  print("STEP 2: Computing support of mu")
  # First part: Compute zeros of the derivative of G_mu inverse
  a1 = mu1.T_tilde(-r_A + epsilon)
  b1 = mu1.T_tilde(r_A - epsilon)
  a2 = mu2.T_tilde(-r_A + epsilon)
  b2 = mu2.T_tilde(r_A - epsilon)
  
  a0 = max(numpy.real(a1), numpy.real(a2))
  b0 = min(numpy.real(b1), numpy.real(b2))
  tol = 10**(-12) 
  
  a_prod_cd = find_zero_derivative_t(a0, a0*epsilon, mu1, mu2, tol)
  b_prod_cd = find_zero_derivative_t(b0*epsilon, b0, mu1, mu2, tol)

  # Second part: Actually compute support of mu
  x1 = mu1.T_inverse(a_prod_cd)
  y1 = mu1.T_inverse(b_prod_cd)
  x2 = mu2.T_inverse(a_prod_cd)
  y2 = mu2.T_inverse(b_prod_cd)
  a_prod = numpy.real(x1*x2* a_prod_cd / (1 + a_prod_cd))
  b_prod = numpy.real(y1*y2* b_prod_cd / (1 + b_prod_cd))

  print("-- Computed support of mu: [", a_prod, ", ", b_prod, "]")
  mutimes = sm.spectral_measure()
  mutimes.set_support(a_prod, b_prod)
  
  # STEP 3: Evaluating T_mu on a suitable curve
  print("STEP 3: Evaluating T_mu on a suitable curve")
  
  # Compute small circle inside and T^{-1}(circle)
  print("-- Computing orange circle")
  # Compute the largest circle that fits inside Gamma1 and Gamma2
  r_B = min(min(abs(numpy.array(Gamma1.discretization))), min(abs(numpy.array(Gamma2.discretization))))
  r_B = min(r_B, min(abs(a_prod_cd), abs(b_prod_cd)))
  r_B = (1-epsilon) * r_B
  Cprod = curves.circle(1, N)
  # Shrink the circle in order to get it inside the region where G is invertible
  for i in range(Cprod.n_points+1):
    A = Cprod.discretization[i] * r_B
    if (abs(numpy.imag(A)) < 0.00001):
      continue
    a1 = mu1.T_inverse(A)
    a2 = mu2.T_inverse(A)
    a_prod2 = a1 * a2 * A / (1+A)
    if (numpy.imag(a_prod2) * numpy.imag(A) > 0): 
      # it means that the largest point does not work and we need to restrict the interval
      # Binary search
      tol = 0.001
      C = (1-epsilon)*A # here I am assuming that C is inside the analiticity region
      while (abs(A-C) > tol):
        B = (A+C)/2
        b1 = mu1.T_inverse(B)
        b2 = mu2.T_inverse(B)
        b_prod2 = b1 * b2 * B / (B+1)
        if (numpy.imag(b_prod2) * numpy.imag(B) > 0):
          A = B
        else:
          C = B
      r_B = abs(B)
  
  print("-- Finished binary search on the radius ")
  # Now resize the circle, such that it is inside the analiticity region
  Cprod = curves.circle(r_B * 0.9, N)
  
  # Compute preimage of Csum with the sum  
  T1invCprod = []
  T2invCprod = []
  for i in range(N+1):
    T1invCprod.append(mu1.T_inverse(Cprod.discretization[i]))
    T2invCprod.append(mu2.T_inverse(Cprod.discretization[i]))
  T1invCprod = curves.curve(N, numpy.array(T1invCprod))
  T2invCprod = curves.curve(N, numpy.array(T2invCprod))
  
  # Product  
  res = T1invCprod.discretization * T2invCprod.discretization * Cprod.discretization / (1 + Cprod.discretization)
  TinvCprod = curves.curve(N, res)
  
  #Choose ellipse outside the small circle Cprod and extract info on mu from it
  print("-- Choosing green circle A")
  # First of all, choose an elliptical contour OUTSIDE the GinvCsum
  # Compute maximum "radius" corresponding to GinvCsum = minimum radius of JinvGinvCsum
  JinvTinvCprod = curves.curve(N, numpy.zeros(N + 1, dtype=complex))
  for i in range(N+1):
    JinvTinvCprod.discretization[i] = mutimes.joukowski_inverse(TinvCprod.discretization[i])
  r_C = min(abs(numpy.array(JinvTinvCprod.discretization))) * (1 - epsilon) 
  print("-- Radius of green circle: ", r_C)
  
  # Choose a suitable number of points for the discretization of the green circle
  M = max(400, 2 * round(8 * numpy.log(10) / numpy.log(1/r_C)))
  print("-- M = ", M, ", m = ", m)  
  JA = curves.ellipse(a_prod, b_prod, r_C, M)
  
  # Evaluate T_mu on the JA using Cauchy integral,
  # knowing that GinvCsum is sent by G_mu into Csum
  # (remember that Csum has radius small_radius)
  print("-- Evaluation of T_mu on green ellipse")
  A = curves.circle(r_C, M)
  w1 = Cprod.discretization[range(0, N)]
  w2 = JinvTinvCprod.discretization[range(0, N)] 
  w3 = []
  for i in range(0, N):
    w3 = numpy.append(w3, derivative_T_inverse_prod(w1[i], mu1, mu2))
  TA = []
  w4 = mutimes.joukowski_inverse_prime(TinvCprod.discretization[range(0, N)])
  for i in range(0, M):
    der = sum((w1**2) * w3 * w4 / (w2 - A.discretization[i]))/N
    TA = numpy.append(TA, der)
  TA = numpy.append(TA, TA[0])  
  TA = curves.curve(M,TA)
  
  # STEP 4: Recover density of mu from its T-transform
  print("STEP 4: Recover density of mu from its T-transform")
  # Get discretization of measure mu  
  unitDisk = curves.circle(1, M)
  
  # Get discretization of measure mu and plot it
  v1 = TA.discretization[1 : M+1]
  v1 = numpy.flip(v1)
  v2 = scipy.fft.ifft(v1)
  v2 = numpy.concatenate((numpy.array([0]), v2[1:m], numpy.zeros(len(v2)-m-1)))
  v3 = numpy.array(v2 * numpy.concatenate((numpy.array([0]), 
  numpy.array(numpy.geomspace(1/r_C, (1/r_C)**(m-1), num=m-1)), 
  numpy.zeros(len(v2)-m))))
  v3 = numpy.append(v3, 0)
  v4 = scipy.fft.fft(v3)
  approx_mu = v4[0:round((len(v2)+1)/2)+1]
  approx_mu = -numpy.imag(approx_mu)/numpy.pi
  alpha = (b_prod - a_prod)/2
  beta = (b_prod + a_prod)/2
  t = numpy.linspace(0, numpy.pi, len(approx_mu))
  t = numpy.cos(t) * alpha + beta
  approx_mu = approx_mu / t
  approx_mu = abs(approx_mu) 
  
  t_coefficients = numpy.array(v3[1:m])
  t_coefficients = numpy.transpose(t_coefficients)
  density_prod = numpy.vectorize(lambda t: 1/numpy.pi * numpy.imag(t_coefficients @ ((mutimes.joukowski_inverse(t))**range(1,m))) / t)
  mutimes.set_density(density_prod)
  
  # Do all the plots
  if (do_plots == 1):
    
    for i in [1,2,3,4,5,6,7,8,9,12]:
      plt.subplot(4, 3, i)
      plt.axis('equal')
      plt.grid(True, which='both')
      plt.axhline(y=0, color='k')
      plt.axvline(x=0, color='k')
    
    plt.subplot(4, 3, 2)
    J1C = curves.ellipse(mu1.a, mu1.b, r_A, N1)
    J1C.plot('blue')
    T1invCprod.plot('orange')
    
    plt.subplot(4, 3, 1)
    J1invT1invCprod = T1invCprod
    for i in range(J1invT1invCprod.n_points+1):
      J1invT1invCprod.discretization[i] = mu1.joukowski_inverse(J1invT1invCprod.discretization[i])
    J1invT1invCprod.plot('orange')
    unitCircle = curves.circle(1, N)
    unitCircle.plot('red')
    C = curves.circle(1-epsilon, N)
    C.plot('blue')
    
    plt.subplot(4, 3, 3)
    Gamma1.plot('blue')
    Cprod.plot('orange')
    
    plt.subplot(4, 3, 5)
    J2C = curves.ellipse(mu2.a, mu2.b, r_A, N2)
    J2C.plot('blue')
    T2invCprod.plot('orange')
    
    plt.subplot(4, 3, 4)
    J2invT2invCprod = T2invCprod
    for i in range(J2invT2invCprod.n_points+1):
      J2invT2invCprod.discretization[i] = mu2.joukowski_inverse(J2invT2invCprod.discretization[i])
    J2invT2invCprod.plot('orange')
    unitCircle.plot('red')
    C.plot('blue')
    
    plt.subplot(4, 3, 6)
    Gamma2.plot('blue')
    Cprod.plot('orange')
    
    plt.subplot(4, 3, 8)
    plt.plot(a_prod, 0, 'ro')
    plt.plot(numpy.real(b_prod), 0, 'ro')
    TinvCprod.plot('orange')
    JA.plot('green')
    
    plt.subplot(4, 3, 7)
    unitCircle.plot('red')
    # Plot A
    A.plot('green')
    # Plot Gamma
    JinvTinvCprod.plot('orange')
    
    plt.subplot(4, 3, 9)
    plt.plot(numpy.real(a_prod_cd), 0, 'ro')
    plt.plot(numpy.real(b_prod_cd), 0, 'ro')
    TA.plot('green')
    Cprod.plot('orange')
    
    plt.subplot(4, 3, 11)
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.plot(t, numpy.minimum(1, abs(approx_mu)), 'red', label = 'free product')
    mu1.plot('blue', 'mu_1')
    mu2.plot('green', 'mu_2')
    plt.legend()
    
    plt.subplot(4, 3, 12)
    Gamma1.plot('blue')
    Gamma2.plot('blue')
    Cprod.plot('orange')

  return [a_prod, b_prod, t, approx_mu, mutimes]
