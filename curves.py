import cmath
import math
import numpy
import matplotlib.pyplot as plt

# A discretized curve in the complex plane (n_points = number of discretization points)
# "discretization" is an array of length "n_points+1" in which the last point coincides with the first one
class curve:
  def __init__(self, n_points, discretization):
    self.n_points = n_points
    self.discretization = numpy.array(discretization)
  
  def plot(self, col):
    plt.plot(numpy.real(self.discretization), numpy.imag(self.discretization), col)

# Ellipse which is the image of a circle of radius "radius" via a Joukowski mapping,
# and then is rescaled such that the foci are in {a, b}  
class ellipse(curve):
  def __init__(self, a, b, radius, n_points):
    self.a = a
    self.b = b
    self.radius = radius

    self.n_points = n_points
    theta = numpy.array(numpy.linspace(0, 2*math.pi, n_points+1))
    zs = numpy.array(radius*numpy.cos(theta)) + 1j*radius*numpy.array(numpy.sin(theta))
    zs = (zs + 1/zs)/2
    zs = zs * (b-a)/2 + (b+a)/2
    self.discretization = zs
    
  def plot(self, col):
    plt.plot(numpy.real(self.discretization), numpy.imag(self.discretization), col)
    plt.plot(numpy.array([self.a, self.b]), numpy.array([0, 0]), 'r')
    plt.plot(self.a, 0, 'ro')
    plt.plot(self.b, 0, 'ro')
      

# Circle with center (0,0) and radius "radius"
class circle(curve):
  def __init__(self, radius, n_points):
    self.radius = radius
    self.n_points = n_points
    theta = numpy.array(numpy.linspace(0, 2*math.pi, n_points+1))
    self.discretization = numpy.array(self.radius*numpy.cos(theta)) + 1j*self.radius*numpy.array(numpy.sin(theta))






