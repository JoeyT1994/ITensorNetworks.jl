using ITensorNetworks: BilinearFormNetwork, norm_sqr_network, BeliefPropagationCache, environment, update_factor,
    disjoint_union
using ITensors: ITensor, inds, contract, dag, prime, replaceinds, commoninds, noprime
using ITensorNetworks.ITensorsExtensions: map_eigvals

function default_update_seq(ψ::AbstractITensorNetwork)
    seq = collect(vertices(ψ))
    return vcat(seq, reverse(seq[1:(length(seq) - 1)]))
end

function bp_inserter(ϕAψ_bpc::BeliefPropagationCache, ψψ_bpc::BeliefPropagationCache, ψ::ITensorNetwork, v;
    cache_update_kwargs)

    messages = environment(ψψ_bpc, partitionvertices(ψψ_bpc, [(v, "ket")]))
    inv_mts =
        map_eigvals.(
        (inv,), messages, first.(inds.(messages)), last.(inds.(messages)); ishermitian = true
        )
    envs = environment(ϕAψ_bpc, [(v, "ket")])
    b = contract(envs; sequence = "automatic")
    local_state_dag = normalize(contract([b; inv_mts]; sequence = "automatic"))
    local_state = dag(copy(local_state_dag))
    for inv_mt in inv_mts
        local_state = replaceinds(local_state, commoninds(local_state, inv_mt), noprime(commoninds(local_state, inv_mt)))
    end

    ψ[v] = copy(local_state)
    ψψ_bpc = update_factor(ψψ_bpc, (v, "ket"), copy(local_state))
    ψψ_bpc = update_factor(ψψ_bpc, (v, "bra"), local_state_dag)
    ψ, ψψ_bpc = normalize(ψ, ψψ_bpc; cache_update_kwargs)

    ϕAψ_bpc = update_factor(ϕAψ_bpc, (v, "ket"), copy(ψ[v]))
    ϕAψ_bpc = update(ϕAψ_bpc; cache_update_kwargs...)

    return ψ, ψψ_bpc, ϕAψ_bpc
end

function optimise(ϕ::ITensorNetwork, A::ITensorNetwork, ψ::ITensorNetwork; cache_update_kwargs = (; maxiter = 20, tol = 1e-8),
    niters::Int64=5)
    ψψ = norm_sqr_network(ψ)
    ψψ_bpc = BeliefPropagationCache(ψψ, group(v -> last(first(first(v))), vertices(ψψ)))
    ψ, ψψ_bpc = normalize(ψ, ψψ_bpc; cache_update_kwargs)

    ϕAψ = disjoint_union("operator" => A,"bra" => dag(ϕ),"ket" => ψ)
    ϕAψ_bpc = BeliefPropagationCache(ϕAψ, group(v -> last(first(first(v))), vertices(ϕAψ)))
    ϕAψ_bpc = update(ϕAψ_bpc; cache_update_kwargs...)

    update_seq = default_update_seq(ψ)

    for i in 1:niters
        for v in update_seq
            ψ, ψψ_bpc, ϕAψ_bpc = bp_inserter(ϕAψ_bpc, ψψ_bpc, ψ, v; cache_update_kwargs)
            @show scalar(ϕAψ_bpc)
        end
    end
end