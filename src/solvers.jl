abstract type HJBMethod end
struct Explicit <: HJBMethod end
struct Implicit <: HJBMethod
    τ::Float64 # Incomplete LU factorization coefficient
end
Implicit() = Implicit(0.1) # default constructor

"""
    HJBIterator(Δ, method)

Defines the solution method for evaluating V_{t-Δ} given V_{t}, A_{t} and R_{t}.

Arguments:
- `Δ`: time step
- `method = [Implicit, Explicit]`: whether to use implicit or explicit solution method.
"""
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

function step!(V0::AbstractArray, V1::AbstractArray, data::HJBData, problem::HJBProblem, iterator::HJBIterator)
    A = iszero(problem.Aexog) ? data.transition : data.transition + problem.Aexog
    return step!(V0, V1, data.reward, A, problem.ρ, iterator)
end

"""
    invariant_value_function(Vinit, problem, hjb_method)

Solve for the invariant value function of `problem`.
"""
function invariant_value_function(
        Vinit::AbstractArray,
        problem::HJBProblem,
        hjb_method::HJBIterator;
        maxiter = 1000,
        tol = 1.0e-12,
        verbose = false,
    )

    data = HJBData(Vinit)
    function iterate!(V0, V1)
        solve_reward_transition!(data, V1, problem)
        step!(V0, V1, data, problem, hjb_method)
        return V0
    end

    V0, V1 = deepcopy(Vinit), deepcopy(Vinit)
    err = Inf
    iter = 0
    while (err > tol) && (iter <= maxiter)
        iterate!(V0, V1)
        err = supnorm(V0, V1)
        copy!(V1, V0)
        verbose && @info "Iteration: $iter; Error: $err"
        iter += 1
    end
    status = err <= tol ? :converged : :max_iterations
    return (value = V0, data = data, iter = iter, err = err, status = status)
end
