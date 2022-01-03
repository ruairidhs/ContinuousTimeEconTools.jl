module UpwindDifferences

#=
Problem:
    ρv(x) = max_c {r(x, c) + ∂vₓ(x)ẋ(x, c)}
    where:  ẋ(x, c) = g(x, c)
            ẋ(x̲, c) ≥ 0
            ẋ(x̅, c) ≤ 0

Let b(x, ∂v) = argmax_c {r(x, c) + ∂vₓ(x)g(x, c)}

This problem can be efficient solved without the use of a cache.
However the cached implementation allows for vectorization.

Inputs:
    1. x -> a one-dimensional state grid, assumed to be evenly spaced, i.e. a LinRange
    2. n -> x.len

Edge cases:
    - Non standard indexing? Enforce LinRange
    - Small n? 
=#

function diff(v, dx, i)
    return (v[i+1] - v[i]) / dx
end

function diff!(dv, v, dx, n)
    for i in 2:n
        dv[i-1] = (v[i] - v[i-1]) / dx
    end
    return dv
end

function updatepolicy!(bs, x, dv, n, policy) 
    bf, bb = bs
    for i in 1:n-1
        bf[i] = policy(x[i], dv[i])
        bb[i] = policy(x[i+1], dv[i])
    end
    return bs
end

function upwind!(R, GF, GB, v, xs, reward, drift, policy, zerodrift)
    dx = step(xs)
    n  = length(xs)
    z  = zero(eltype(GB))

    # Lower state constraint
    dvf = diff(v, dx, 1)
    bf = policy(xs[1], dvf)
    gf = drift(xs[1], bf)
    if gf > z
        R[1] = reward(xs[1], bf)
        GF[1] = gf
    else
        R[1] = reward(xs[1], zerodrift(xs[1]))
        GF[1] = z
    end
    GB[1] = z

    # State interior
    for i in 2:n-1
        dvb = dvf
        dvf = diff(v, dx, i)
        R[i], GF[i], GB[i] = get_interior_upwind(xs[i], dvf, dvb, reward, drift, policy, zerodrift)
    end

    # Upper state constraint
    dvb = diff(v, dx, n-1)
    bb = policy(xs[end], dvb)
    gb = drift(xs[end], bb)
    if gb < z
        R[end] = reward(xs[end], bb)
        GB[end] = gb
    else
        R[end] = reward(xs[end], zerodrift(xs[end]))
        GB[end] = z
    end
    GF[end] = z

    return R, GF, GB
end

function get_interior_upwind(x, dvf, dvb, reward, drift, policy, zerodrift)
    # Kernel of upwinding algorithm 
    bf = policy(x, dvf)
    bb = policy(x, dvb) 
    gf = drift(x, bf)
    gb = drift(x, bb)
    z = zero(typeof(gf)) 

    if gf > 0 && gb ≥ 0 
        R = reward(x, bf)
        GF = gf
        GB = z
    elseif gb < 0 && gf ≤ 0
        R = reward(x, bb)
        GF = z
        GB = gb
    elseif gf ≤ 0 && gb ≥ 0
        R = reward(x, zerodrift(x))
        GF = z
        GB = z
    else
        rf = reward(x, bf)
        Hf = rf + dvf * gf
        rb = reward(x, bb)
        Hb = rb + dvb * gb
        H0 = reward(x, zerodrift(x))

        if Hf > max(Hb, H0)
            R = rf
            GF = gf
            GB = z 
        elseif Hb > H0
            R = rb
            GF = z 
            GB = gb
        else
            R = H0
            GF = z
            GB = z 
        end 
    end
    return R, GF, GB
end

function upwind!(R, GF, GB, bs, dv, v, x, reward, drift, policy, zerodrift)
    bf, bb = bs
    dx = step(x)
    n = length(x) 

    diff!(dv, v, dx, n)
    updatepolicy!(bs, x, dv, n, policy)

    gf = drift(x[1], bf[1])
    if gf > zero(eltype(GF))
        R[1] = reward(x[1], bf[1])
        GF[1] = gf
    else
        R[1] = reward(x[1], zerodrift(x[1]))
        GF[1] = zero(eltype(GF))
    end
    GB[1] = zero(eltype(GB))

    for i in 2:n-1
        R[i], GF[i], GB[i] = get_interior_upwind(x[i], dv[i], dv[i-1], reward, drift, policy, zerodrift)
        #=
        gf = drift(x[i], bf[i])
        gb = drift(x[i], bb[i-1])
        if gf > 0 && gb ≥ 0
            R[i] = reward(x[i], bf[i])
            GF[i] = gf
            GB[i] = zero(eltype(GB))
        elseif gb < 0 && gf ≤ 0
            R[i] = reward(x[i], bb[i-1])
            GF[i] = zero(eltype(GF))
            GB[i] = gb
        elseif gf ≤ 0 && gb ≥ 0
            R[i] = reward(x[i], zerodrift(x[i]))
            GF[i] = zero(eltype(GF))
            GB[i] = zero(eltype(GB))
        else
            rf = reward(x[i], bf[i])
            Hf = rf + dv[i] * gf
            rb = reward(x[i], bb[i-1])
            Hb = rb + dv[i-1] * gb
            H0 = reward(x[i], zerodrift(x[i]))

            if Hf > max(Hb, H0)
                R[i] = rf
                GF[i] = gf
                GB[i] = zero(eltype(GB))
            elseif Hb > H0
                R[i] = rb
                GF[i] = zero(eltype(GF))
                GB[i] = gb
            else
                R[i] = H0
                GF[i] = zero(eltype(GF))
                GB[i] = zero(eltype(GB))
            end
        end
        =#
    end
    
    gb = drift(x[end], bb[end])
    if gb < 0.0
        R[end] = reward(x[end], bb[end])
        GB[end] = gb
    else
        R[end] = reward(x[end], zerodrift(x[end]))
        GB[end] = zero(eltype(GB))
    end
    GF[end] = zero(eltype(GF))

    return R, GF, GB
end

export upwind!

end # module