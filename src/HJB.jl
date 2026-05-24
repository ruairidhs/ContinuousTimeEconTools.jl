"""
    HJBProblem

Defines the parameters of a HJB problem.
"""
struct HJBProblem{T <: Real, F0, F1, F2, F3, V <: AbstractVector, M <: AbstractMatrix}
    ρ::T
    reward::F0
    policy::F1
    drift::F2
    zerodrift::F3
    x::V
    Aexog::M
end

"""
    HJBData

A data cache containing all arrays required to compute a HJB iteration.
Can be re-used across iterations.
"""
struct HJBData{U <: Upwinder, V <: AbstractArray, T <: Tridiagonal}
    upwinder::U
    value::V
    reward::V
    drift::V
    transition::T
end

function HJBData(V::AbstractArray)
    N = length(V)
    value = similar(V)
    reward = similar(V)
    drift = similar(V)

    fi = first(outer_indices(V))
    upwinder = Upwinder(size(V, 1), view(reward, :, fi...), view(drift, :, fi...))
    transition = Tridiagonal(zeros(N - 1), zeros(N), zeros(N - 1))
    return HJBData(upwinder, value, reward, drift, transition)
end

function solve_reward_transition!(
        data::HJBData, V::AbstractArray, problem::HJBProblem
    )
    nx = length(problem.x)
    # Use upwinding to produce the reward vector and endogenous transition matrix
    loc = 0 # tracker for the flat indices in the transition matrix
    for inds in outer_indices(V)
        get_view(a) = view(a, :, inds...)
        # Specialize each function to this value of the exogenous state
        funcs = (
            (x, c) -> problem.reward(x, c, inds...),
            (x, dv) -> problem.policy(x, dv, inds...),
            (x, c) -> problem.drift(x, c, inds...),
            x -> problem.zerodrift(x, inds...),
        )
        # Run upwinding on the correct slice of data.reward and data.drift
        set_reward!(data.upwinder, get_view(data.reward))
        set_drift!(data.upwinder, get_view(data.drift))
        data.upwinder(get_view(V), problem.x, funcs)
        # Update the endogenous state transition matrix
        inds = (loc + 1):(loc + nx)
        @views policy_matrix!(data.transition.dl[(loc + 1):(loc + nx - 1)], data.transition.d[inds], data.transition.du[(loc + 1):(loc + nx - 1)], problem.x, data.upwinder)
        loc += nx
    end
    return data
end
