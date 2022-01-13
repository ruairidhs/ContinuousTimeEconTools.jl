# UpwindDifferences

This is a package for implementing a finite differences method based on an upwind scheme to solve Hamilton-Jacobi-Bellman (HJB) equations.

In particular, this package can be used to solve the following problem:

$$
\rho v(x) = \max_{c} r(x, c) + \frac{\partial v}{\partial x}(x)\dot{x}(x,c)
$$
