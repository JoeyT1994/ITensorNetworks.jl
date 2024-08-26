using ITensorNetworks: BilinearFormNetwork, norm_sqr_network, BeliefPropagationCache, environment, update_factor
using ITensors: ITensor, inds, contract, dag, prime, replaceinds, commoninds, noprime
using ITensorNetworks.ITensorsExtensions: map_eigvals

function default_update_seq(ψ::AbstractITensorNetwork)
    seq = collect(vertices(ψ))
    return vcat(seq, reverse(seq[1:(length(seq) - 1)]))
end

function bp_updater(ϕAψ_bpc::BeliefPropagationCache, v)
    envs = environment(ϕAψ_bpc, [(v, "ket")])
    return dag(contract(envs); sequence = "automatic")
end

function bp_inserter(local_state::ITensor, ϕAψ_bpc::BeliefPropagationCache, ψψ_bpc::BeliefPropagationCache, ψ::ITensorNetwork, v;
    cache_update_kwargs)

    messages = environment(ψψ_bpc, partitionvertices(ψψ_bpc, [(v, "ket")]))
    inv_sqrt_mts =
        map_eigvals.(
        (inv ∘ sqrt,), messages, first.(inds.(messages)), last.(inds.(messages)); ishermitian = true
        )
    local_state = contract([local_state; inv_sqrt_mts]; sequence = "automatic")
    local_state = contract([local_state; inv_sqrt_mts]; sequence = "automatic")
    local_state = normalize(local_state)

    l_inds = setdiff(inds(local_state), uniqueinds(ψ, v))
    dag_local_state = dag(replaceinds(local_state, l_inds, l_inds'))

    ψ[v] = copy(local_state)
    ψψ_bpc = update_factor(ψψ_bpc, (v, "ket"), local_state)
    ψψ_bpc = update_factor(ψψ_bpc, (v, "bra"), dag_local_state)
    ψ, ψψ_bpc = normalize(ψ, ψψ_bpc; cache_update_kwargs)

    ϕAψ_bpc = update_factor(ϕAψ_bpc, (v, "ket"), copy(ψ[v]))
    ϕAψ_bpc = update(ϕAψ_bpc; cache_update_kwargs...)

    return ψ, ψψ_bpc, ϕAψ_bpc
end

function optimise(ψ::ITensorNetwork, ϕAψ::BilinearFormNetwork; cache_update_kwargs = (; maxiter = 20, tol = 1e-8),
    niters::Int64=5)
    ψψ = norm_sqr_network(ψ)
    ψψ_bpc = BeliefPropagationCache(ψψ, group(v -> last(first(first(v))), vertices(ψψ)))
    ψ, ψψ_bpc = normalize(ψ, ψψ_bpc; cache_update_kwargs)
    ϕAψ_bpc = BeliefPropagationCache(ϕAψ, group(v -> last(first(first(v))), vertices(ϕAψ)))
    ϕAψ_bpc = update(ϕAψ_bpc; cache_update_kwargs...)

    update_seq = default_update_seq(ψ)

    for i in 1:niters
        for v in update_seq
            new_state = bp_updater(ϕAψ_bpc, v)
            ψ, ψψ_bpc, ϕAψ_bpc = bp_inserter(new_state, ϕAψ_bpc, ψψ_bpc, ψ, v; cache_update_kwargs)
            @show scalar(ϕAψ_bpc)
        end
    end
end