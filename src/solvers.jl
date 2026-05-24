abstract type HJBMethod end
struct Explicit <: HJBMethod end
struct Implicit <: HJBMethod
    τ::Float64 # Incomplete LU factorization coefficient
end
Implicit() = Implicit(0.1) # default constructor

struct HJBIterator{T, M} <: HJBMethod
    Δ::T
    method::M
end

function step!(v0, v1, r, A, ρ, iterator::HJBIterator{T, Implicit}) where {T}
    v0, v1, r = vec(v0), vec(v1), vec(r)
    b = r .+ (1 / iterator.Δ) .* v1 # if doing iterative solution the allocations are irrelevant for runtime
    Q = (ρ + 1 / iterator.Δ) * I - A
    p = ilu(Q, τ = iterator.method.τ)
    v0 .= v1 # use v1 as an initial guess but don't want to overwrite it
    IterativeSolvers.bicgstabl!(v0, Q, b; Pl = p)
    return v0
end

function step!(v0, v1, r, A::Tridiagonal, ρ, iterator::HJBIterator{T, Implicit}) where {T}
    # one-dimensional case: can use fast tridiagonal solve
    v0, v1, r = vec(v0), vec(v1), vec(r)
    v0 .= r .+ (1 / iterator.Δ) .* v1
    ldiv!(factorize((ρ + 1 / iterator.Δ) * I - A), v0)
    return v0
end

function step!(v0, v1, r, A, ρ, iterator::HJBIterator{T, Explicit}) where {T}
    v0, v1, r = vec(v0), vec(v1), vec(r)
    v0 .= r
    mul!(v0, (ρ * I - A), v1, 1, -1)
    # now v0 contains (1/Δ)(v1 - v)
    v0 .*= -iterator.Δ
    v0 .+= v1
    return v0
end

function step!(data::HJBData, V::AbstractArray, problem::HJBProblem, iterator::HJBIterator)
    A = iszero(problem.Aexog) ? data.transition : data.transition + problem.Aexog
    return step!(data.value, V, data.reward, A, problem.ρ, iterator)
end

function invariant_value_function(
        Vinit::AbstractArray,
        problem::HJBProblem,
        hjb_method::HJBIterator;
        maxiter = 1000,
        tol = 1.0e-12,
        verbose = false,
    )

    data = HJBData(Vinit)
    function iterate!(V)
        solve_reward_transition!(data, V, problem)
        step!(data, V, problem, hjb_method)
        return data.value
    end

    V = deepcopy(Vinit)
    err = Inf
    iter = 0
    while (err > tol) && (iter <= maxiter)
        iterate!(V)
        err = supnorm(data.value, V)
        V .= data.value
        verbose && @info "Iteration: $iter; Error: $err"
        iter += 1
    end
    status = err <= tol ? :converged : :max_iterations
    return (data = data, iter = iter, err = err, status = status)
end


# struct HJBIteratorTerminal{T, M, N} <: HJBMethod
#     ρ::T
#     Δ::T
#     method::M
#     VT::Array{T, N} # terminal value function
#     λ::T # arrival rate of transitioning to the terminal value function
# end
#
# function (HJB::HJBIteratorTerminal{T, Implicit, N})(v0, v1, r, A) where {T, N}
#     v0, v1, vt, r = vec(v0), vec(v1), vec(HJB.VT), vec(r)
#     # saves allocations compared to just setting r = r + λ .* vt
#     v0 .= r .+ (1 / HJB.Δ) .* v1 .+ HJB.λ .* vt
#     ldiv!(factorize((HJB.ρ + HJB.λ + 1 / HJB.Δ) * I - A), v0)
#     return v0
# end
#
# function (HJB::HJBIteratorTerminal{T, Explicit, N})(v0, v1, r, A) where {T, N}
#     v0, v1, vt, r = vec(v0), vec(v1), vec(HJB.VT), vec(r)
#     v0 .= r .+ HJB.λ .* vt
#     mul!(v0, ((HJB.ρ + HJB.λ) * I - A), v1, 1, -1)
#     # now v0 contains (1/Δ)(v1 - v)
#     v0 .*= -HJB.Δ
#     v0 .+= v1
#     return v0
# end
