# ContinuousTimeEconTools

Some functions that I use to solve continuous-time economics models using upwind finite-differences.

# Mathematical formulation

This package solves continuous-time Hamilton-Jacobi-Bellman (HJB)
equations with one continuous state and any number of additional exogenous
state dimensions. Let $x \in \mathbb{R}$ denote the continuous state, let
$y \in \mathcal{Y}$ collect the remaining states, and let $c$ denote the
control. The Hamilton-Jacobi-Bellman equation is:

$$
\rho v(t, x, y)
=
\max_c
\left\{
    r(x, c, y)
    + g(x, c, y) \partial_x v(t, x, y)
\right\}
+ \mathcal{A}_y v(t, x, y)
+ \partial_t v(t, x, y),
$$

where:

- $\rho$ is the discount rate;
- $r(x, c, y)$ is the flow payoff;
- $g(x, c, y)$ is the drift of the continuous state;
- $\mathcal{A}_y$ is the generator for exogenous state transitions, which are
  not affected by the control.

The package discretizes $x$ on a grid and uses an upwind finite-difference
scheme for the controlled drift term. After discretization, the HJB update has
the form

$$
\rho V_t = R_t + A_t V_t + \frac{V_{t+1} - V_t}{\Delta},
$$

where $V_t$ is the vectorized value function, $R_t$ is the vector of flow
payoffs evaluated at the selected controls, and $A_t$ is the transition
generator implied by the upwind drift and any exogenous transition matrix.

# Usage

## HJB

Methods are provided to solve the discrete HJB equation

$$
\rho V_t = R_t + A_t V_t + \frac{V_{t+1} - V_t}{\Delta}.
$$

Given $V_{t+1}$, the package steps backward and computes $V_t$ using either an
implicit or explicit update.

Implicit update:

$$
\left(\rho + \frac{1}{\Delta}\right) V_t - A_t V_t
=
R_t + \frac{1}{\Delta} V_{t+1}.
$$

In the implementation this is a semi-implicit update: $R_t$ and $A_t$ are
computed from the policy and drift implied by $V_{t+1}$, then held fixed while
solving the linear system for $V_t$.

Explicit update:

$$
V_t
=
V_{t+1}
+ \Delta \left(R_t + A_t V_{t+1} - \rho V_{t+1}\right).
$$

## Upwinding

For each point on the $x$ grid, the upwind step evaluates the policy using the
forward and backward finite differences of $V_{t+1}$. It then keeps the
derivative whose implied drift points into the grid cell:

- use the forward derivative when $g(x, c, y) > 0$;
- use the backward derivative when $g(x, c, y) < 0$;
- use the zero-drift control when $g(x, c, y) = 0$.

This produces the payoff vector $R_t$ and the controlled transition generator
$A_t$ used by the HJB update.

## API


