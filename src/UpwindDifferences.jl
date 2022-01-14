"""
    UpwindDifferences

Implements a finite differences method based on an upwind scheme to solve Hamilton-Jacobi-Bellman (HJB) equations.
"""
module UpwindDifferences

#=
Problem:
    ρv(x) = max_c {r(x, c) + ∂vₓ(x)ẋ(x, c)}
    where:  ẋ(x, c) = g(x, c)
            ẋ(x̲, c) ≥ 0
            ẋ(x̅, c) ≤ 0

Let b(x, ∂v) = argmax_c {r(x, c) + ∂vₓ(x)g(x, c)}

This problem can be efficient solved without the use of a cache.
However the cached implementation allows for vectorization.

Inputs:
    1. x -> a one-dimensional state grid, assumed to be evenly spaced, i.e. a LinRange
    2. n -> x.len

Edge cases:
    - Non standard indexing? Enforce LinRange
    - Small n? 
=#

"""Contains the optimal reward vector, `R`, and forward and backward drift, `GF` and `GB`."""
struct UpwindResult{T <: Number}
    R::Vector{T}
    GF::Vector{T}
    GB::Vector{T}
end

"""Compute the forward difference of vector `v` at index i"""
function diff(v, dx, i)
    return (v[i+1] - v[i]) / dx
end

"""
    upwind!(UR::UpwindResult, v, xs, reward, drift, policy, zerodrift)

Update `UR` using upwind finite differences based on discretized value function `v`.

See [`upwind`](@ref) for further details.
`UR` contains 3 vectors: (R) the optimal reward; (GF) the optimal drift when non-negative; (GB) the optimal drift when non-positive.
"""
function upwind!(UR::UpwindResult, v, xs, reward, drift, policy, zerodrift)

    R, GF, GB = UR.R, UR.GF, UR.GB
    dx = step(xs)
    n  = length(xs)
    z  = zero(eltype(GB))

    # Lower state constraint
    dvf = diff(v, dx, 1)
    bf = policy(xs[1], dvf)
    gf = drift(xs[1], bf)
    if gf > z
        R[1] = reward(xs[1], bf)
        GF[1] = gf
    else
        R[1] = reward(xs[1], zerodrift(xs[1]))
        GF[1] = z
    end
    GB[1] = z

    # State interior
    for i in 2:n-1
        dvb = dvf
        dvf = diff(v, dx, i)
        R[i], GF[i], GB[i] = get_interior_upwind(xs[i], dvf, dvb, reward, drift, policy, zerodrift)
    end

    # Upper state constraint
    dvb = diff(v, dx, n-1)
    bb = policy(xs[end], dvb)
    gb = drift(xs[end], bb)
    if gb < z
        R[end] = reward(xs[end], bb)
        GB[end] = gb
    else
        R[end] = reward(xs[end], zerodrift(xs[end]))
        GB[end] = z
    end
    GF[end] = z

    return UR
end

"""
    upwind(v::Vector, xs::LinRange, reward, drift, policy, zerodrift) 
    
Compute the optimal reward, and forward and backward drift, based on value function `v` using an upwind differences scheme.

This function can be used when an upwind finite differences approximation of the value function derivative is required to compute an optimal control solution.
As the resulting drift is required to compute the upwind approximation, it is also returned.

See also [`upwind!`](@ref) for an efficient inplace version.

# Arguments
- `v::Vector`: the discretized value function from which the result will be computed.
- `xs::LinRange`: the (one-dimensional) discretized state-space at which `v` is evaluated.
- `reward(x, c)::Function`: returns the flow reward given state value `x` and control value `c`.
- `drift(x, c)::Function`: returns the drift in the state given current state value `x` and control value `v`.
- `policy(x, dv)::Function`: returns the optimal control value given current state value `x` and value function derivative `dv`.
- zerodrift(x)::Function`: returns the control value which results in zero drift, i.e, `drift(x, zerodrift(x)) = 0`.
"""
function upwind(v, xs, reward, drift, policy, zerodrift)
    UR = UpwindResult(similar(v), similar(v), similar(v))
    upwind!(UR, v, xs, reward, drift, policy, zerodrift)
    return UR
end

function get_interior_upwind(x, dvf, dvb, reward, drift, policy, zerodrift)
    # Kernel of upwinding algorithm 
    bf = policy(x, dvf)
    bb = policy(x, dvb) 
    gf = drift(x, bf)
    gb = drift(x, bb)
    z = zero(typeof(gf)) 

    if gf > 0 && gb ≥ 0 
        R = reward(x, bf)
        GF = gf
        GB = z
    elseif gb < 0 && gf ≤ 0
        R = reward(x, bb)
        GF = z
        GB = gb
    elseif gf ≤ 0 && gb ≥ 0
        R = reward(x, zerodrift(x))
        GF = z
        GB = z
    else
        rf = reward(x, bf)
        Hf = rf + dvf * gf
        rb = reward(x, bb)
        Hb = rb + dvb * gb
        H0 = reward(x, zerodrift(x))

        if Hf > max(Hb, H0)
            R = rf
            GF = gf
            GB = z 
        elseif Hb > H0
            R = rb
            GF = z 
            GB = gb
        else
            R = H0
            GF = z
            GB = z 
        end 
    end
    return R, GF, GB
end


export UpwindResult, upwind!, upwind

end # module
