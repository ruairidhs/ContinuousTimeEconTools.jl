# Test that it produces the same results as an older version of the code
include("old_upwinding.jl")

# ==== One dimensional case ====
# Use CRRA utility and positive interest rate
function make_HJB_functions(θ, y)
    @assert θ > 1.0
    minc, maxc = 1e-4, 1e4
    reward(_, c) = (c ^ (1 - θ) - 1) / (1 - θ)
    function policy(_, dv::T) where {T}
        dv <= zero(T) && return maxc
        base_c = dv ^ (-1/θ)
        return max(minc, min(maxc, base_c))
    end
    drift(x, c) = 0.02 * x + y - c
    zd(x) = 0.02 * x + y
    return (reward, policy, drift, zd)
end

function test_grid(xg, HJBfuncs, ρ)
    vinit = @. (1 / ρ) * log(xg + 1.0)
    nx = length(xg)
    reward, policy, drift, zd = HJBfuncs

    # Run the new and old methods and check the reward and transitions are equal
    U = ContinuousTimeEconTools.Upwinder(xg)
    U(vinit, xg, HJBfuncs)
    Anew = Tridiagonal(zeros(nx-1), zeros(nx), zeros(nx-1))
    ContinuousTimeEconTools.policy_matrix!(Anew, xg, U)

    old_res = OldCode.upwind(vinit, xg, reward, drift, policy, zd)
    Aold = Tridiagonal(zeros(nx-1), zeros(nx), zeros(nx-1))
    OldCode.policy_matrix!(Aold, old_res)

    @test maximum(abs.(old_res.R ./ U.rf .- 1)) .< 1e-12
    @test all((old_res.GF .== 0) .== (max.(U.gf, 0) .== 0))
    @test all((old_res.GB .== 0) .== (min.(U.gb, 0) .== 0))
    @test maximum(abs.(Aold .- Anew)) .< 1e-12

    # Run the same check but with a non-concave initial value function
    vinit_non_concave = max.(log.(xg .+ 0.01), 0.1 .* log.(xg .+ 0.01) .- 0.2)

    U = ContinuousTimeEconTools.Upwinder(xg)
    U(vinit_non_concave, xg, HJBfuncs)
    Anew = Tridiagonal(zeros(nx-1), zeros(nx), zeros(nx-1))
    ContinuousTimeEconTools.policy_matrix!(Anew, xg, U)

    old_res = OldCode.upwind(vinit_non_concave, xg, reward, drift, policy, zd)
    Aold = Tridiagonal(zeros(nx-1), zeros(nx), zeros(nx-1))
    OldCode.policy_matrix!(Aold, old_res)

    @test maximum(abs.(old_res.R ./ U.rf .- 1)) .< 1e-12
    @test all((old_res.GF .== 0) .== (max.(U.gf, 0) .== 0))
    @test all((old_res.GB .== 0) .== (min.(U.gb, 0) .== 0))
    @test maximum(abs.(Aold .- Anew)) .< 1e-12

    # check that if I run a backwards iteration step I get the same
    Anew .= 0
    vres0, vres1 = copy(vinit), copy(vinit)
    r, g = ones(nx), ones(nx)
    U = ContinuousTimeEconTools.Upwinder(xg, view(r, :, ()...), view(g, :, ()...))
    for iter in 1:10
        Anew .= 0
        ContinuousTimeEconTools.backwards_iterate!(vres0, vres1, xg, U, r, g, Anew, HJBfuncs,
                                ContinuousTimeEconTools.HJBIterator(0.05, 10.0, ContinuousTimeEconTools.Implicit())
                               )
        vres1 .= vres0
    end

    vres0_old, vres1_old = copy(vinit), copy(vinit)
    rcache = similar(vres0_old)
    for iter in 1:10
        or = OldCode.upwind(vres1_old, xg, reward, drift, policy, zd)
        rcache .= or.R
        OldCode.policy_matrix!(Aold, or)
        vres0_old .= or.R .+ (1 / 10.0) .* vres1_old
        ldiv!(factorize((0.05 + 1 / 10.0) * I - Aold), vres0_old)
        vres1_old .= vres0_old
    end

    @test all(U.rf .≈ rcache)
    @test all(Anew .≈ Aold)
    @test all(vres1 .≈ vres1_old)
    @test all(OldCode.extract_drift(Aold, xg) .≈ g)
end

test_grid(range(0.0, 1.0, length = 100), make_HJB_functions(2.0, 1.0), 0.05)
test_grid(vcat([0.0], exp.(range(log(1e-6), log(1), length = 100))), make_HJB_functions(2.0, 1.0), 0.05)

# ===== 2d dimensional test =====
function make_Λy(ys)
    base = rand(length(ys), length(ys)) .* 0.1
    base[diagind(base)] .= 0.0
    for i in axes(base, 1)
        base[i, i] = -sum(base[i, :])
    end
    return base
end

function make_HJB_functions_2d(θ, ys)
    @assert θ > 1.0
    minc, maxc = 1e-4, 1e4
    reward(_, c, _) = (c ^ (1 - θ) - 1) / (1 - θ)
    function policy(_, dv::T, _) where {T}
        dv <= zero(T) && return maxc
        base_c = dv ^ (-1/θ)
        return max(minc, min(maxc, base_c))
    end
    drift(x, c, yi) = 0.02 * x + ys[yi] - c
    zd(x, yi) = 0.02 * x + ys[yi]
    return (reward, policy, drift, zd)
end

function test_grid_2d(xg, ys, Λy, HJBfuncs, ρ)
    vinit = @. (1 / ρ) * log(xg + $permutedims(ys))
    nx, ny = length(xg), length(ys)
    reward, policy, drift, zd = HJBfuncs
    Δt = 10.0

    # new
    R, G = similar(vinit), similar(vinit)
    A = ContinuousTimeEconTools.make_exogenous_transition(nx, [Λy])
    U = ContinuousTimeEconTools.Upwinder(xg, view(R, :, 1), view(G, :, 1))
    HJB = ContinuousTimeEconTools.HJBIterator(0.05, Δt, ContinuousTimeEconTools.Implicit())
    vres0, vres1 = copy(vinit), copy(vinit)
    ContinuousTimeEconTools.backwards_iterate!(vres0, vres1, xg, U, R, G, A, HJBfuncs, HJB)

    # old
    Aexog = kron(sparse(Λy), I(nx))
    Aendog = kron(I(ny), sparse(Tridiagonal(zeros(nx-1), zeros(nx), zeros(nx-1))))
    Rold = zeros(nx * ny)
    vres0_old, vres1_old = copy(vinit), copy(vinit)
    function iterate_old!(v0, v1)
        loc = 0
        for yi in axes(v1, 2)
            ur = OldCode.upwind(view(v1, :, yi), xg,
                                (x, c) -> reward(x, c, yi),
                                (x, c) -> drift(x, c, yi),
                                (x, dv) -> policy(x, dv, yi),
                                x -> zd(x, yi),
                               )
            inds = loc+1:loc+nx
            Rold[inds] .= ur.R
            OldCode.policy_matrix!(@view(Aendog[inds, inds]), ur)
            loc += nx
        end
        vec(v0) .= Rold .+ (1 / Δt) .* vec(v1)
        ldiv!(factorize((ρ + 1 / Δt) * I - (Aexog + Aendog)), vec(v0))
    end
    iterate_old!(vres0_old, vres1_old)

    # Check for equality
    @test typeof(A) <: SparseMatrixCSC
    @test vec(R) ≈ Rold
    @test all(Aexog .== ContinuousTimeEconTools.make_exogenous_transition(nx, [Λy]))
    @test all((Aexog + Aendog) .≈ A)
    @test vres0 ≈ vres0_old

    # and now check the invariant
    new_res = ContinuousTimeEconTools.invariant_value_function(vinit, xg, Aexog, HJBfuncs, HJB)
    for iter in 1:500
        iterate_old!(vres0_old, vres1_old)
        vres1_old .= vres0_old
    end
    @test maximum(abs.(vres1_old .- new_res.value)) .<= 1e-10
end

ys = [0.25, 0.5, 0.75, 1.0]
Λy = make_Λy(ys)
test_grid_2d(range(0.0, 1.0, length = 100), ys, Λy, make_HJB_functions_2d(2.0, ys), 0.05)
test_grid_2d(vcat([0.0], exp.(range(log(1e-6), log(1), length=100))), ys, Λy, make_HJB_functions_2d(2.0, ys), 0.05)
