outer_indices(V::AbstractVector) = Base.Iterators.repeated((), 1)
outer_indices(A::AbstractArray{T,N}) where {T,N} =
    Iterators.product((axes(A, i) for i = 2:N)...)

function backwards_iterate!(V0, V1, x, U::Upwinder, R, G, A, funcs, HJB::HJBIterator)
    reward, policy, drift, zerodrift = funcs
    nx = length(x)
    loc = 0
    for inds in outer_indices(V1)
        inner_funcs = ((x, c) -> reward(x, c, inds...),
                 (x, dv) -> policy(x, dv, inds...),
                 (x, c) -> drift(x, c, inds...),
                 x -> zerodrift(x, inds...),
                )
        set_reward!(U, view(R, :, inds...))
        set_drift!(U, view(G, :, inds...))
        U(view(V1, :, inds...), x, inner_funcs)
        inds = loc+1:loc+nx
        policy_matrix!(@view(A[inds, inds]), x, U)
        loc += nx
    end
    HJB(V0, V1, R, A)
end

function backwards_iterate!(V0, V1, VT, x, U::Upwinder, R, G, A, λ, funcs, HJB::HJBIterator)
    reward, policy, drift, zerodrift = funcs
    nx = length(x)
    loc = 0
    for inds in outer_indices(V1)
        inner_funcs = ((x, c) -> reward(x, c, inds...),
                 (x, dv) -> policy(x, dv, inds...),
                 (x, c) -> drift(x, c, inds...),
                 x -> zerodrift(x, inds...),
                )
        set_reward!(U, view(R, :, inds...))
        set_drift!(U, view(G, :, inds...))
        U(view(V1, :, inds...), x, inner_funcs)
        inds = loc+1:loc+nx
        policy_matrix!(view(A, inds, inds), x, U)
        loc += nx
    end
    HJB(V0, V1, VT, R, A, λ)
end 

function invariant_value_function(Vinit, x, Aexog, funcs, HJB::HJBIterator; fixedpoint_kwargs...)
    R = similar(Vinit)
    G = similar(Vinit)
    fi = first(outer_indices(Vinit))
    U = Upwinder(x, view(R, :, fi...), view(G, :, fi...))
    A = deepcopy(Aexog)
    function iterate!(V0, V1)
        A .= Aexog # clean the transition matrix prior to each iteration
        backwards_iterate!(V0, V1, x, U, R, G, A, funcs, HJB)
    end
    fp_res = fixedpoint(iterate!, Vinit; fixedpoint_kwargs...)
    return (value = fp_res.value,
            transition = A,
            R = R, G = G,
            iter = fp_res.iter,
            err = fp_res.err,
            status = fp_res.status
           )
end

function invariant_value_function(Vinit, VT, x, Aexog, λ, funcs, HJB::HJBIterator; fixedpoint_kwargs...)
    R = similar(Vinit)
    G = similar(Vinit)
    fi = first(outer_indices(Vinit))
    U = Upwinder(x, view(R, :, fi...), view(G, :, fi...))
    A = deepcopy(Aexog)
    function iterate!(V0, V1)
        A .= Aexog # clean the transition matrix prior to each iteration
        backwards_iterate!(V0, V1, VT, x, U, R, G, A, λ, funcs, HJB)
    end
    fp_res = fixedpoint(iterate!, Vinit; fixedpoint_kwargs...)
    return (value = fp_res.value,
            transition = A,
            R = R, G = G,
            iter = fp_res.iter,
            err = fp_res.err,
            status = fp_res.status
           )
end

function make_exogenous_transition(nx, Λs)
    foldl((acc, M) -> kron(I(size(M, 1)), acc) + kron(sparse(M), I(size(acc, 1))),
          Λs; init = spzeros(nx, nx))
end
