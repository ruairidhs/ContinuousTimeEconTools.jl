"""
    supnorm(u, v)
Find the largest element-wise difference between two collections.
"""
function supnorm(u, v)
    return maximum(abs.(u .- v))
end

"""
    fixedpoint(iterate!, init[, cache]; distance=supnorm, copy_func = copy!, maxiter=1000, tol=1e-10)
Repeatedly call iterate!, starting with `init`, until an approximate fixed point is reached.

Two iterations must be stored in order to calculate the distance between operations.
These iterations are stored in `cache`, which can be optionally supplied to avoid allocations.

An alternative `copy_func` may be specified for cases where `copy!(x1, x0)` is not sufficient to copy values into `x1`.

# Examples
```jldoctest
julia> init = [1.0, 2.0];
julia> cache = (similar(init), similar(init));
julia> function iterate!(x1, x0, β)
           x1 .= β .* x0
       end;
julia> fixedpoint((x1, x0) -> iterate!(x1, x0, 0.9), init, cache)
(value = [4.165953662869907e-10, 8.331907325739814e-10], iter = 205, err = 9.257674806377564e-11, status = :converged)
```
"""
function fixedpoint(
    iterate!,
    init,
    cache;
    distance = supnorm,
    copy_func = copy!,
    maxiter = 1000,
    tol = 1e-12,
    verbose = false,
    err_increase_tol = 0.0,
)
    x0, x1 = cache
    copy_func(x0, init)
    err = Inf
    err_old = Inf

    for iter = 1:maxiter
        iterate!(x1, x0)
        err = distance(x1, x0)
        verbose && @info "Iteration: $iter; Error: $err"
        err - err_old >= err_increase_tol &&
            return (value = x0, iter = iter, err = err, status = :error_increase)
        err <= tol && return (value = x1, iter = iter, err = err, status = :converged)
        copy_func(x0, x1)
        err_old = err
    end
    return (value = x0, iter = maxiter, err = err, status = :max_iterations)
end

function fixedpoint(iterate!, init; kwargs...)
    cache = (copy(init), copy(init))
    return fixedpoint(iterate!, init, cache; kwargs...)
end
