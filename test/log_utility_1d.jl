# Note only works for ρ = r so state constraint has no impact
function make_HJB_functions(r)
    reward(_, c) = log(c)
    policy(_, dv) = 1 / dv
    drift(x, c) = r * x - c
    zerodrift(x) = r * x
    return (reward, policy, drift, zerodrift)
end

function get_analytic_value(xgrid, ρ, r)
    cons = log(ρ) + (r / ρ) - 1
    return (1 / ρ) .* (log.(xgrid) .+ cons)
end

function get_numeric_value(xgrid, ρ, r, Δ, method)
    HJBfuncs = make_HJB_functions(r)
    vinit = collect(xgrid)
    nx = length(xgrid)
    A = Tridiagonal(zeros(nx - 1), zeros(nx), zeros(nx - 1))
    res = invariant_value_function(vinit, xgrid, A, HJBfuncs,
                                   HJBIterator(ρ, Δ, method);
                                   maxiter = 1_000_000,
                                   err_increase_tol = 1.0,
                                   verbose=false,
                                  )
    res.status == :converged || error("failed to converge")
    return res.value
end

function get_spec_error(spec)
    (; xgrid, ρ, r, Δ, method) = spec
    av = get_analytic_value(xgrid, ρ, r)
    nv = get_numeric_value(xgrid, ρ, r, Δ, method)
    err = maximum(abs, (av ./ nv .- 1) .* 100)
    return err
end

specs = [(xgrid = range(0.01, 1.0, length = 25), ρ = 0.05, r = 0.05, Δ = 10.0, method = Implicit()),
         (xgrid = range(0.01, 1.0, length = 100), ρ = 0.05, r = 0.05, Δ = 10.0, method = Implicit()),
         (xgrid = range(0.01, 1.0, length = 1000), ρ = 0.05, r = 0.05, Δ = 10.0, method = Implicit()),
         (xgrid = range(0.01, 1.0, length = 10_000), ρ = 0.05, r = 0.05, Δ = 10.0, method = Implicit()),
         (xgrid = range(0.0001, 10.0, length = 1000), ρ = 0.05, r = 0.05, Δ = 10.0, method = Implicit()),
        ]

for spec in specs
    @test get_spec_error(spec) < 0.01
end

