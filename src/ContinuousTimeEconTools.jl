"""
    ContinuousTimeEconTools

Implements a finite differences method based on an upwind scheme to solve Hamilton-Jacobi-Bellman (HJB) equations.
"""
module ContinuousTimeEconTools

using LinearAlgebra

#=
Problem:
    ρv(x) = max_c {r(x, c) + ∂vₓ(x)ẋ(x, c)}
    where:  ẋ(x, c) = g(x, c)
            ẋ(x̲, c) ≥ 0
            ẋ(x̅, c) ≤ 0

Let b(x, ∂v) = argmax_c {r(x, c) + ∂vₓ(x)g(x, c)}
=#

include("upwind.jl")
include("HJB.jl")
include("utils.jl")

export UpwindResult, 
       upwind!, upwind,
       empty_policy_matrix,
       policy_matrix, policy_matrix!,
       extract_drift, extract_drift!,
       fixedpoint,
       iterateHJB!, iterateHJBVI!

end # module
