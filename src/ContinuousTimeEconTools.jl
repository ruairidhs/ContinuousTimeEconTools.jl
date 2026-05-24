"""
    ContinuousTimeEconTools

Implements a finite differences method based on an upwind scheme to solve Hamilton-Jacobi-Bellman (HJB) equations.
"""
module ContinuousTimeEconTools

using LinearAlgebra, SparseArrays, LoopVectorization, IterativeSolvers, IncompleteLU

include("utils.jl")
include("upwind.jl")
include("hjb.jl")
include("solvers.jl")

export HJBData, HJBProblem, HJBIterator, Implicit, Explicit, invariant_value_function, make_exogenous_transition, solve_reward_transition!

end # module
