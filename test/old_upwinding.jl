# regression test against the old code
module OldCode
using LinearAlgebra

struct UpwindResult{T<:Number}
    R::Vector{T}
    GF::Vector{T}
    GB::Vector{T}
end

UpwindResult(v::AbstractVector) =
    UpwindResult(map(v -> zeros(eltype(v), length(v)), (v, v, v))...)
UpwindResult(T, n::Int) = UpwindResult(map(v -> zeros(T, n), 1:3)...)

function fdiff(v, dx, i)
    return (v[i+1] - v[i]) / dx
end

function upwind!(UR::UpwindResult, v, xs, reward, drift, policy, zerodrift)

    Base.require_one_based_indexing(v, xs)

    R, GF, GB = UR.R, UR.GF, UR.GB
    n = length(xs)
    n >= 3 || throw(ArgumentError("state space must be larger than 3 points"))
    n == length(v) ||
        throw(ArgumentError("length of value function must equal length of state-space"))
    z = zero(eltype(GB))

    # Lower state constraint: asserts that min(drift)=0
    dxf = xs[2] - xs[1]
    dvf = fdiff(v, dxf, 1)
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
    for i = 2:n-1
        dvb = dvf
        dxf = xs[i+1] - xs[i]
        dvf = fdiff(v, dxf, i)
        R[i], GF[i], GB[i] =
            get_interior_upwind(xs[i], dvf, dvb, reward, drift, policy, zerodrift)
    end

    # Upper state constraint: asserts that max(drift)=0
    dvb = fdiff(v, dxf, n - 1)
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

    # finally, scale the forward and backward drifts by the xsteps
    dx = xs[2] - xs[1]
    GF[1] /= dx
    for i = 2:n-1
        GB[i] /= dx
        dx = xs[i+1] - xs[i]
        GF[i] /= dx
    end
    GB[n] /= dx

    return UR
end

function upwind(v, xs, reward, drift, policy, zerodrift)
    UR = UpwindResult(v)
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

    if gf > 0 && gb ≥ 0 # forward difference
        R = reward(x, bf)
        GF = gf
        GB = z
    elseif gb < 0 && gf ≤ 0 # backward difference
        R = reward(x, bb)
        GF = z
        GB = gb
    elseif gf ≤ 0 && gb ≥ 0 # zero change
        R = reward(x, zerodrift(x))
        GF = z
        GB = z
    else # convex value function
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

function policy_matrix!(A, UR)
    A[diagind(A, -1)] .= .-@view(UR.GB[2:end])
    A[diagind(A, 0)] .= UR.GB .- UR.GF
    A[diagind(A, 1)] .= @view(UR.GF[1:end-1])
    return A
end

function extract_drift!(drifts, A, xs)
    length(drifts) == length(xs) ||
        throw(ArgumentError("lengths of drifts and xs do not match"))
    dx = diff(xs) # TODO allocates
    drifts .= zero(eltype(drifts))
    drifts[begin:end-1] .= diag(A, 1) .* dx
    drifts[begin+1:end] .-= diag(A, -1) .* dx
    return drifts
end

function extract_drift(A, xs)
    drifts = zeros(length(xs))
    return extract_drift!(drifts, A, xs)
end
end #module
