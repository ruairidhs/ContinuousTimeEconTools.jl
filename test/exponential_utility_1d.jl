# Test the 1d solution for u(c) = -(1/θ) * exp(-θc)
# We have an analytic solution for the policy function to compare against
function get_analytic_policy(xgrid, θ, y, ρ)
    return y .+ sqrt.(2 * (ρ / θ) * xgrid)
end

function define_problem(x, θ, y, ρ)
    minc, maxc = 1.0e-4, 1.0e4
    reward(_, c) = -(1 / θ) * exp(-θ * c)
    function policy(_, dv::T) where {T}
        dv <= zero(T) && return maxc
        base_c = -(1 / θ) * log(dv)
        return max(minc, min(maxc, base_c))
    end
    drift(_, c) = y - c
    zd(_) = y

    nx = length(x)
    Aexog = Tridiagonal(zeros(nx - 1), zeros(nx), zeros(nx - 1))
    return HJBProblem(ρ, reward, policy, drift, zd, x, Aexog)
end


function get_numerical_policy(x, θ, y, ρ, Δ, method)
    Vinit = @. (1 / ρ) * log(x .+ 1.0)
    problem = define_problem(x, θ, y, ρ)
    hjb_method = HJBIterator(Δ, method)
    out = invariant_value_function(Vinit, problem, hjb_method, maxiter = 1_000_000, tol = 1.0e-12)
    out.status == :converged || error("failed to converge")
    return map(problem.drift, x, out.data.drift)
end

function get_spec_error(xgrid, θ, y, ρ, Δ, method)
    ap = get_analytic_policy(xgrid, θ, y, ρ)
    np = get_numerical_policy(xgrid, θ, y, ρ, Δ, method)
    err = maximum(abs.((np ./ ap .- 1) .* 100))
    return err
end

specs = [
    (range(0.0, 1.0, length = 25), 2.0, 1.0, 0.05), # different grid sizes
    (range(0.0, 1.0, length = 100), 2.0, 1.0, 0.05),
    (range(0.0, 1.0, length = 500), 2.0, 1.0, 0.05),
    (range(0.0, 1.0, length = 1000), 2.0, 1.0, 0.05),
    (vcat([0.0], exp.(range(log(1.0e-6), log(1), length = 100))), 2.0, 1.0, 0.05), # irregular grid
    (range(0.0, 1.0, length = 100), 2.0, 1.0, 0.1), # different parameters
    (range(0.0, 1.0, length = 100), 2.0, 1.0, 0.01),
    (range(0.0, 1.0, length = 100), 2.0, 10.0, 0.05),
    (range(0.0, 1.0, length = 100), 8.0, 1.0, 0.05),
]
for spec in specs
    @test get_spec_error(spec..., 1000.0, Implicit()) < 0.1 # 0.1% error
end

# Briefly test the implicit method: requires many iterations!
@test get_spec_error(specs[2]..., 0.01, Explicit()) < 0.1
