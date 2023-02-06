# Mutable because I want to be able to change rf and gf
mutable struct Upwinder{T,RF<:AbstractVector{T},GF<:AbstractVector{T}}
    N::Int
    rf::RF # results
    gf::GF
    dv::Vector{T}
    rb::Vector{T} # caches
    gb::Vector{T}
    rz::Vector{T}
    If::BitVector
    Ib::BitVector
    Is::BitVector
end

Upwinder(x) = Upwinder(
    length(x),
    similar(x),
    similar(x),
    zeros(eltype(x), length(x) - 1),
    similar(x),
    similar(x),
    similar(x),
    BitVector(false for _ = 1:length(x)),
    BitVector(false for _ = 1:length(x)),
    BitVector(false for _ = 1:length(x)),
)

function Upwinder(x, r, g)
    size(x) == size(r) || throw(DimensionMismatch("x and r have incompatible lengths"))
    size(x) == size(g) || throw(DimensionMismatch("x and g have incompatible lengths"))
    Base.require_one_based_indexing(x, r, g) || throw(ArgumentError("array inputs must have one based indexing"))
    n = length(x)
    return Upwinder(
        n,
        r,
        g,
        zeros(eltype(x), n - 1),
        similar(x),
        similar(x),
        similar(x),
        BitVector(false for _ = 1:n),
        BitVector(false for _ = 1:n),
        BitVector(false for _ = 1:n),
    )
end

function set_reward!(U::Upwinder, r)
    length(r) == U.N || throw(DimensionMismatch())
    U.rf = r
    return nothing
end

function set_drift!(U::Upwinder, g)
    length(g) == U.N || throw(DimensionMismatch())
    U.gf = g
    return nothing
end

# Place the drift into a policy matrix
function policy_matrix!(A, x, U::Upwinder{T, RF, GF}) where {T, RF, GF}
    size(A) == (U.N, U.N) || throw(DimensionMismatch("incompatible A"))
    length(x) == U.N || throw(DimensionMismatch("incompatible x"))
    # First scale by the x-step and sort into forward and backward drifts
    # I want to keep the 'raw' drift in gf, so I'll use rz in place of gf
    F, B = U.rz, U.gb # just caches to be overwritten
    drifts = U.gf
    Z = zero(T)
    for i in 1:U.N
        if drifts[i] > Z
            F[i] = drifts[i] / (x[i+1] - x[i])
            B[i] = Z
        elseif drifts[i] < Z
            B[i] = drifts[i] / (x[i] - x[i-1])
            F[i] = Z
        else
            F[i] = Z
            B[i] = Z
        end
    end
    # Then increment the values in the matrix
    # (which may already contain some exogenous drift components)
    A[diagind(A, -1)] .-= @view(B[2:end])
    A[diagind(A, 0)] .+= B .- F
    A[diagind(A, 1)] .+= @view(F[1:end-1])
    return nothing
end

# Implementation of the upwind finite-differences algorithm
function (U::Upwinder)(v, x, funcs)
    U.N == length(v) || throw(DimensionMismatch("incompatible length(v)"))
    U.N == length(x) || throw(DimensionMismatch("incompatible length(x)"))
    reward, policy, drift, zerodrift = funcs
    (; dv, rf, gf, rb, gb, rz, If, Ib, Is) = U
    # Ensure boundary conditions are satisfied
    If[end] = false; Ib[begin] = false
    gf[end] = zero(eltype(gf)); gb[begin] = zero(eltype(gf))
    rf[end] = zero(eltype(rf)); rb[begin] = zero(eltype(rb))

    dv!(dv, v, x)
    fill_forward!(rf, gf, If, x, dv, reward, policy, drift)
    fill_backward!(rb, gb, Ib, x, dv, reward, policy, drift)
    fill_zero!(rz, x, reward, zerodrift)
    @. begin
        gf += gb
        rf += rb
        Is = !(If | Ib)
        rf += rz * Is
        Is = If & Ib # Hamiltonian flag
    end # sufficient for concave value functions
    convex_points!(rf, gf, Is, x, dv, funcs)
    return nothing
end

function dv!(dv, v, x)
    for i = 1:length(dv)
        dv[i] = (v[i+1] - v[i]) / (x[i+1] - x[i])
    end
    return nothing
end

function fill_forward!(r, g, If, x, dv, reward, policy, drift)
    inds = 1:length(x)-1
    for i in inds
        b = policy(x[i], dv[i])
        gi = drift(x[i], b)
        ri = reward(x[i], b)
        pos = gi > 0
        If[i] = pos
        g[i] = gi * pos
        r[i] = ri * pos
    end
    return nothing
end

function fill_backward!(r, g, Ib, x, dv, reward, policy, drift)
    inds = 2:length(x)
    for i in inds
        b = policy(x[i], dv[i-1])
        gi = drift(x[i], b)
        ri = reward(x[i], b)
        neg = gi < 0
        Ib[i] = neg
        g[i] = gi * neg
        r[i] = ri * neg
    end
    return nothing
end

function fill_zero!(rz, x, reward, zerodrift)
    for (i, x) in enumerate(x)
        rz[i] = reward(x, zerodrift(x))
    end
    return nothing
end

# Functions to deal with non-concave parts of the value function
function hamiltonian_point(x, dvf, dvb, funcs)
    reward, policy, drift, zerodrift = funcs

    bf = policy(x, dvf)
    rf = reward(x, bf)
    gf = drift(x, bf)
    hf = rf + dvf * gf

    bb = policy(x, dvb)
    rb = reward(x, bb)
    gb = drift(x, bb)
    hb = rb + dvb * gb

    rz = reward(x, zerodrift(x))

    if rz >= max(hf, hb) # zero is optimal
        return rz, zero(gf)
    elseif hf >= hb
        return rf, gf
    else
        return rb, gb
    end
end

function convex_points!(r, g, Is, x, dv, funcs)
    i = 1
    while true
        i = findnext(Is, i)
        if isnothing(i)
            return nothing
        else
            ri, gi = hamiltonian_point(x[i], dv[i], dv[i-1], funcs)
            r[i] = ri
            g[i] = gi
            i += 1
        end
    end
end
