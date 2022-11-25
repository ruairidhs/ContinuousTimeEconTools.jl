struct Explicit end
struct Implicit end

"""
    iterateHJB!(v0, v1, r, A, ρ, Δ, method)
    iterateHJB!(v0, v1, vt, r, A, ρ, λ, Δ, method)

Approximate `v0 ≡ v_t` given `v1 ≡ v_{t+Δ}` using the HJB equation: ``ρv = r + A * v + v̇``.
A terminal value `vt` with arrival rate `λ` can optionally be included, in which case the HJB
is ``ρv = r + A * v + v̇ + λ(vt - v)``.

# Methods

Two methods are available to approximate `v̇`.

-  `Implicit()` uses a forwards approximation, i.e,
   the method solves ``ρv0 = r + A * v0 + (1 / Δ) * (v1 - v0)`` for `v0`.
   This requires a matrix division but is valid for large `Δ`.
-  `Explicit()` uses a backwards approximation, i.e.,
   the method solves ``ρv1 = r + A * v1 + (1 / Δ) * (v1 - v0)`` for `v0`.
   This only requires matrix multiplication but is only valid for small `Δ`.
"""
function iterateHJB!(v0, v1, r, A, ρ, Δ, ::Explicit)
    v0 .= r
    mul!(v0, (ρ * I - A), v1, 1, -1)
    # now v0 contains (1/Δ)(v1 - v)
    v0 .*= -Δ
    v0 .+= v1
    return v0
end

function iterateHJB!(v0, v1, vt, r, A, ρ, λ, Δ, ::Explicit)
    v0 .= r .+ λ .* vt
    mul!(v0, ((ρ + λ) * I - A), v1, 1, -1)
    # now v0 contains (1/Δ)(v1 - v)
    v0 .*= -Δ
    v0 .+= v1
    return v0
end

function iterateHJB!(v0, v1, r, A, ρ, Δ, ::Implicit)
    v0 .= r .+ (1 / Δ) .* v1
    ldiv!(factorize((ρ + 1 / Δ) * I - A), v0)
    return v0
end

function iterateHJB!(v0, v1, vt, r, A, ρ, λ, Δ, ::Implicit)
    # saves allocations compared to just setting r = r + λ .* vt
    v0 .= r .+ (1 / Δ) .* v1 .+ λ .* vt
    ldiv!(factorize((ρ + λ + 1 / Δ) * I - A), v0)
    return v0
end

function iterateHJBVI!(v0, v1, va, r, A, ρ, Δ, LCPsolver; tol=1e-2)
    M = (ρ + 1 / Δ) * I - A
    q = -(r .+ v1 ./ Δ) .+ M * va
    v0 .= max.(v1 .- va, 0)
    for _ in 1:20
        v0 .= LCPsolver(v0, M, q)
        maximum(abs.(v0 .* (M * v0 + q))) < tol && break
    end
    v0 .+= va
    return v0
end
iterateHJBVI!(v0, v1, va, vt, r, A, ρ, λ, Δ, LCPsolver; kwargs...) = iterateHJBVI!(v0, v1, va, r .+ λ .* vt, A, ρ + λ, Δ, LCPsolver; kwargs...)

empty_policy_matrix(n::Int) = Tridiagonal(map(zeros, (n-1, n, n-1))...)

"""
    policy_matrix!(A, UR::UpwindResult)

Update tridiagonal band of `A` to the Poisson transition matrix implied by the drifts in `UR`.
"""
function policy_matrix!(A, UR::UpwindResult)
    A[diagind(A, -1)] .= .-@view(UR.GB[2:end]) 
    A[diagind(A, 0)]  .= UR.GB .- UR.GF
    A[diagind(A, 1)]  .= @view(UR.GF[1:end-1])
    return A
end

"""
    policy_matrix(A, UR::UpwindResult)

Return a tridiagonal matrix representation of the Poisson transition matrix implied by the drifts in `UR`.
"""
function policy_matrix(UR::UpwindResult)
    A = empty_policy_matrix(length(UR.R))
    return policy_matrix!(A, UR)
end


function extract_drift!(drifts, A, xs)
    length(drifts) == length(xs) || throw(ArgumentError("lengths of drifts and xs do not match"))
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
