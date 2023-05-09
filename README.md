# Free additive and multiplicative convolution

### 1. Objectives

This code allows you to compute an approximation of the density function associated 
to the free additive or multiplicative convolution of two given measures. In
order for the theory to hold, the two measures need to have the following characteristics:

- Compact support (positive in the case of free multiplicative convolution)
- Have a density function with sufficiently nice behavior at the boundary
- Invertible Cauchy transform (for the additive case) or invertible T-transform (for the multiplicative case)

These assumptions ensure that the resulting measure has a sqrt-behavior at the boundary and that its density 
function is regular enough; see [1] for precise statements.

### 2. Dependencies

The toolbox requires: numpy, scipy, matplotlib, random, math, cmath.

### 3. Reproducing the numerical experiments in the paper

There are some scripts for reproducing the figures in the paper:

- generate_example_images.py: figures in Section 3;
- test_quadrature.py: Figure 5;
- test_recovery.py: Figure 6;
- test_additive_convolution.py and test_multiplicative_convolution.py: all other figures in Section 6.

### 4. Using the toolbox for your own examples

See the scripts demo_sum.py and demo_product.py for examples, and the Jupyter notebook

### 5. References

[1] Alice Cortinovis and Lexing Ying. Computing free convolutions via contour integrals (2023). https://arxiv.org/abs/2305.01819

### 6. Contact

For bugs / questions / feedback: [Contact](mailto:alicecor@stanford.edu)
