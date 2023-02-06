"""
    ContinuousTimeEconTools

Implements a finite differences method based on an upwind scheme to solve Hamilton-Jacobi-Bellman (HJB) equations.
"""
module ContinuousTimeEconTools

using LinearAlgebra, SparseArrays, LoopVectorization

include("upwind.jl")
include("HJB.jl")
include("utils.jl")
include("dimensions.jl")

export backwards_iterate!,
    invariant_value_function,
    make_exogenous_transition,
    Upwinder,
    HJBIterator,
    Explicit,
    Implicit

end # module
