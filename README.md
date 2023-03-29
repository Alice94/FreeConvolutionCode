# Toolbox for free additive or multiplicative convolution

## 1. Objectives

This code allows you to compute an approximation of the density function associated 
to the free additive or multiplicative convolution of two given measures. In
order for the theory to hold, the two measures need to have the following characteristics:

- Compact support
- Have a density function with nice behavior at the boundary
- Invertible Cauchy transform (for the additive case) or invertible T-transform (for the multiplicative case)
- E[X] nonzero in the case of free multiplication

These assumptions ensure that the resulting measure has a sqrt-behavior at the boundary and that its density 
function is regular enough.

## 2. Dependencies

The toolbox requires: numpy, scipy, matplotlib, random, math, cmath

(Are they all necessary?)

## 3. Reproducing the numerical experiments in the paper

TODO

## 4. Using the toolbox for your own examples

See the scripts demo_sum.py and demo_product.py for examples, and the Jupyter notebook

## 5. References

TODO
