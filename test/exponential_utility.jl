@testset "Exponential utility" begin

    function make_HJB_functions(θ, y)
        minc, maxc = 1e-4, 1e4
        reward(x, c) = -(1 / θ) * exp(-θ * c)
        drift(x, c) = y - c
        function policy(x, dv)
            dv <= 0 && return maxc
            base_c = -(1 / θ) * log(dv)
            return max(minc, min(maxc, base_c))
        end
        zerodrift(x) = y
        return (reward, drift, policy, zerodrift)
    end

    function solve_HJB(xgrid, ρ, θ, y, dt, method, maxiter)
        n = length(xgrid)
        vinit = @. (1 / ρ) * log(xgrid .+ 1.0)
        UR = UpwindResult(vinit)
        A = sparse(Tridiagonal(zeros(n - 1), zeros(n), zeros(n - 1)))
        reward, drift, policy, zerodrift = make_HJB_functions(θ, y)

        function iterate!(v0, v1, dt)
            upwind!(UR, v1, xgrid, reward, drift, policy, zerodrift)
            policy_matrix!(A, UR)
            iterateHJB!(v0, v1, UR.R, A, ρ, dt, method)
            return v0
        end

        res = fixedpoint(
            (v0, v1) -> iterate!(v0, v1, dt),
            vinit;
            maxiter = maxiter,
            verbose = false,
            err_increase_tol = 1.0,
        )
        return (value = res.value, A = A, status = res.status)
    end

    function get_policy(A, xgrid, θ, y)
        _, drift, _, _ = make_HJB_functions(θ, y)
        drifts = extract_drift(A, xgrid)
        return map(drift, xgrid, drifts) # consumption at each grid point
    end

    function analytic_policy(xgrid, θ, y, ρ)
        return y .+ sqrt.(2 * (ρ / θ) * xgrid)
    end

    function test_spec(xgrid, θ, y, ρ, method, dt, maxiter)
        value_res = solve_HJB(xgrid, ρ, θ, y, dt, method, maxiter)
        @test value_res.status == :converged
        numerical_policy_res = get_policy(value_res.A, xgrid, θ, y)
        analytic_policy_res = analytic_policy(xgrid, θ, y, ρ)
        rel_errs = (numerical_policy_res ./ analytic_policy_res .- 1) .* 100
        max_err = maximum(abs.(rel_errs))
        #@info "Maximum error: $max_err"
        @test max_err <= 0.1 # maximum 0.1% error
    end

    # Test the explicit method
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
        test_spec(spec..., ContinuousTimeEconTools.Implicit(), 1000.0, 1000)
    end

    # Briefly test the implicit method: requires many iterations!
    test_spec(specs[2]..., ContinuousTimeEconTools.Explicit(), 0.01, 100000)
end
