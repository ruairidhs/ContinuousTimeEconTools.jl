function make_HJB_functions(θ, y)
    minc, maxc = 1e-4, 1e4
    reward(_, c) = -(1 / θ) * exp(-θ * c)
    #policy(_, dv::T) where {T} = dv <= zero(T) ? maxc : max(minc, -(1 / θ) * log(dv))
    function policy(x, dv::T) where {T}
        dv <= zero(T) && return maxc
        base_c = -(1 / θ) * log(dv)
        return max(minc, min(maxc, base_c))
    end
    drift(_, c) = y - c
    zd(_) = y
    return (reward, policy, drift, zd)
end

function get_analytic_policy(xgrid, θ, y, ρ)
    return y .+ sqrt.(2 * (ρ / θ) * xgrid)
end

function get_numerical_policy(xgrid, θ, y, ρ, Δ, method)
    HJBfuncs = make_HJB_functions(θ, y)
    vinit = @. (1 / ρ) * log(xgrid .+ 1.0)
    nx = length(xgrid)
    A = Tridiagonal(zeros(nx-1), zeros(nx), zeros(nx-1))
    res = invariant_value_function(vinit, xgrid, A, HJBfuncs,
                                   HJBIterator(ρ, Δ, method);
                                   maxiter = 1_000_000, err_increase_tol = 1.0,
                                   verbose = false
                                  )
    res.status == :converged || error("failed to converge")
    return map(HJBfuncs[3], xgrid, res.G)
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
    (vcat([0.0], exp.(range(log(1e-6), log(1), length = 100))), 2.0, 1.0, 0.05), # irregular grid
    (range(0.0, 1.0, length = 100), 2.0, 1.0, 0.10), # different parameters
    (range(0.0, 1.0, length = 100), 2.0, 1.0, 0.01),
    (range(0.0, 1.0, length = 100), 2.0, 10.0, 0.05),
    (range(0.0, 1.0, length = 100), 8.0, 1.0, 0.05),
]
for spec in specs
    @test get_spec_error(spec..., 1000.0, Implicit()) < 0.1 # 0.1% error
end

# Briefly test the implicit method: requires many iterations!
@test get_spec_error(specs[2]..., 0.01, Explicit()) < 0.1
