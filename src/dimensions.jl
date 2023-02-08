outer_indices(V::AbstractVector) = Base.Iterators.repeated((), 1)
outer_indices(A::AbstractArray{T,N}) where {T,N} =
    Iterators.product((axes(A, i) for i = 2:N)...)

function backwards_iterate!(
    V0,
    V1,
    x,
    U::Upwinder,
    R,
    G,
    A::Tridiagonal,
    Ax,
    funcs,
    HJB::HJBMethod,
)
    reward, policy, drift, zerodrift = funcs
    nx = length(x)
    loc = 0
    for inds in outer_indices(V1)
        inner_funcs = (
            (x, c) -> reward(x, c, inds...),
            (x, dv) -> policy(x, dv, inds...),
            (x, c) -> drift(x, c, inds...),
            x -> zerodrift(x, inds...),
        )
        set_reward!(U, view(R, :, inds...))
        set_drift!(U, view(G, :, inds...))
        U(view(V1, :, inds...), x, inner_funcs)
        inds = loc+1:loc+nx
        @views policy_matrix!(A.dl[loc+1:loc+nx-1], A.d[inds], A.du[loc+1:loc+nx-1], x, U)
        loc += nx
    end
    HJB(V0, V1, R, iszero(Ax) ? A : A + Ax) # if Ax is zero then we can use Tridiagonal exact solve
end

function invariant_value_function(
    Vinit,
    x,
    Aexog,
    funcs,
    HJB::HJBMethod;
    fixedpoint_kwargs...,
)
    R = similar(Vinit)
    G = similar(Vinit)
    fi = first(outer_indices(Vinit))
    U = Upwinder(x, view(R, :, fi...), view(G, :, fi...))
    A = Tridiagonal(zeros(length(R) - 1), zeros(length(R)), zeros(length(R) - 1)) # this has to be zeros #TODO assertion
    function iterate!(V0, V1)
        backwards_iterate!(V0, V1, x, U, R, G, A, Aexog, funcs, HJB)
    end
    fp_res = fixedpoint(iterate!, Vinit; fixedpoint_kwargs...)
    return (
        value = fp_res.value,
        transition = A,
        R = R,
        G = G,
        iter = fp_res.iter,
        err = fp_res.err,
        status = fp_res.status,
    )
end

function make_exogenous_transition(nx, Λs)
    foldl(
        (acc, M) -> kron(I(size(M, 1)), acc) + kron(sparse(M), I(size(acc, 1))),
        Λs;
        init = spzeros(nx, nx),
    )
end
