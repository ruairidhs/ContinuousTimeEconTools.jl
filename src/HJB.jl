abstract type HJBMethod end
struct Explicit <: HJBMethod end
struct Implicit <: HJBMethod
    τ::Float64 # Incomplete LU factorization coefficient
end
Implicit() = Implicit(0.10) # default constructor

struct HJBIterator{T,M} <: HJBMethod
    ρ::T
    Δ::T
    method::M
end

function (HJB::HJBIterator{T,Implicit})(v0, v1, r, A) where {T}
    v0, v1, r = vec(v0), vec(v1), vec(r)
    b = r .+ (1 / HJB.Δ) .* v1 # if doing iterative solution the allocations are irrelevant for runtime
    Q = (HJB.ρ + 1 / HJB.Δ) * I - A
    p = ilu(Q, τ = HJB.method.τ)
    v0 .= v1 # use v1 as an initial guess but don't want to overwrite it
    IterativeSolvers.bicgstabl!(v0, Q, b; Pl = p)
    return v0
end

function (HJB::HJBIterator{T,Implicit})(v0, v1, r, A::Tridiagonal) where {T}
    # one-dimensional case: can use fast tridiagonal solve
    v0, v1, r = vec(v0), vec(v1), vec(r)
    v0 .= r .+ (1 / HJB.Δ) .* v1
    ldiv!(factorize((HJB.ρ + 1 / HJB.Δ) * I - A), v0)
    return v0
end

function (HJB::HJBIterator{T,Explicit})(v0, v1, r, A) where {T}
    v0, v1, r = vec(v0), vec(v1), vec(r)
    v0 .= r
    mul!(v0, (HJB.ρ * I - A), v1, 1, -1)
    # now v0 contains (1/Δ)(v1 - v)
    v0 .*= -HJB.Δ
    v0 .+= v1
    return v0
end

struct HJBIteratorTerminal{T,M,N} <: HJBMethod
    ρ::T
    Δ::T
    method::M
    VT::Array{T,N} # terminal value function
    λ::T # arrival rate of transitioning to the terminal value function
end

function (HJB::HJBIteratorTerminal{T,Implicit,N})(v0, v1, r, A) where {T,N}
    v0, v1, vt, r = vec(v0), vec(v1), vec(HJB.VT), vec(r)
    # saves allocations compared to just setting r = r + λ .* vt
    v0 .= r .+ (1 / HJB.Δ) .* v1 .+ HJB.λ .* vt
    ldiv!(factorize((HJB.ρ + HJB.λ + 1 / HJB.Δ) * I - A), v0)
    return v0
end

function (HJB::HJBIteratorTerminal{T,Explicit,N})(v0, v1, r, A) where {T,N}
    v0, v1, vt, r = vec(v0), vec(v1), vec(HJB.VT), vec(r)
    v0 .= r .+ HJB.λ .* vt
    mul!(v0, ((HJB.ρ + HJB.λ) * I - A), v1, 1, -1)
    # now v0 contains (1/Δ)(v1 - v)
    v0 .*= -HJB.Δ
    v0 .+= v1
    return v0
end
