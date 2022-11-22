struct Explicit end
struct Implicit end

"""
    backwards_iterate!(v0, v1, r, A, ρ, Δ, method)

Approximate `v0 ≡ v_t` given `v1 ≡ v_{t+Δ}` using the HJB equation: ``ρv = r + A * v + v̇``.

# Methods

Two methods are available:

-  `ExplicitBackwardsIteration()` uses a backwards approximation of the time derivative, i.e.,
   the method solves: ``ρv1 = r + A * v1 + (1 / Δ) * (v1 - v0)``;
-  `ImplicitBackwardsIteration()` uses a forwards approximation of the time derivative, i.e,
   the method solves: ``ρv0 = r + A * v0 + (1 / Δ) * (v1 - v0)``.
"""
function iterateHJB!(v0, v1, r, A, ρ, Δ, ::Explicit)
    v0 .= r
    mul!(v0, (ρ * I - A), v1, 1, -1)
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
    n = length(UR.R)
    A = Tridiagonal(map(zeros, (n-1, n, n-1))...)
    return policy_matrix!(A, UR)
end
