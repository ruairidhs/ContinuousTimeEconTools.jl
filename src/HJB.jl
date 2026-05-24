"""
    HJBProblem

Defines the parameters of a HJB problem.

# Arguments

- `ρ`: discount rate
- `reward::Callable`: evaluate the flow reward given the state and control
- `policy::Callable`: evaluate the optimal control given the state and gradient of value function wrt. the endogenous state
- `drift::Callable`: evaluate the endogenous state drift given the state and control
- `zerodrift::Callable`: evaluate the control which results in zero drift given the state.
- `x::AbstractVector`: the grid of the endogenous state
- `Aexog::AbstractMatrix`: the transition matrix of the exogenous state
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
    HJBData(V::AbstractArray)

A data cache used for upwinding.

# Usage
- Construction depends on the dimensions and data type (not values) of the example value function.
- Fields are filled by `solve_reward_transition!` for a given value function and `HJBProblem`.

# Fields

- `upwinder::U`: internal cache used for computation
- `reward`: the flow reward at the optimal policy
- `drift`: the drift of the endogenous state at the optimal policy
- `transition`: the transition matrix of the endogenous state under the optimal policy
"""
struct HJBData{U <: Upwinder, V <: AbstractArray, T <: Tridiagonal}
    upwinder::U
    reward::V
    drift::V
    transition::T
end

function HJBData(V::AbstractArray)
    N = length(V)
    reward = similar(V)
    drift = similar(V)

    fi = first(outer_indices(V))
    upwinder = Upwinder(size(V, 1), view(reward, :, fi...), view(drift, :, fi...))
    transition = Tridiagonal(zeros(N - 1), zeros(N), zeros(N - 1))
    return HJBData(upwinder, reward, drift, transition)
end

"""
    solve_reward_transition!(data::HJBData, V::AbstractArray, problem::HJBProblem)

Compute the optimal reward, drift and endogenous transition matrix given `V` and store in `data`.

This function does not allocate.
"""
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
