using ITensorNetworks: BilinearFormNetwork, norm_sqr_network, BeliefPropagationCache, environment, update_factor,
    disjoint_union
using ITensors: ITensor, inds, contract, dag, prime, replaceinds, commoninds, noprime
using ITensorNetworks.ITensorsExtensions: map_eigvals
using ITensorNetworks: contraction_sequence
using OMEinsumContractionOrders
using Statistics

function default_update_seq(ψ::AbstractITensorNetwork; nsites::Int = 1)
    if nsites == 1
        seq = collect(vertices(ψ))
        return vcat(seq, reverse(seq[1:(length(seq) - 1)]))
    else
        seq = edges(ψ)
        return vcat(seq, reverse(reverse.(seq[1:(length(seq) - 1)])))
    end    
end

function default_update_seq_two(ψ::AbstractITensorNetwork)
    seq = collect(vertices(ψ))
    return vcat(seq, reverse(seq[1:(length(seq) - 1)]))
end

function bp_extracter(ψ::ITensorNetwork, A::ITensorNetwork, ϕ, v; cache_update_kwargs)

    ψ = norm_orthogonalize(ψ; v)
    ϕAψ_bpc = build_sandwich(ψ, A, ϕ)
    seq = post_order_dfs_edges(partitioned_graph(ϕAψ_bpc), parent(only(partitionvertices(ϕAψ_bpc, [(v, "ket")]))))
    ϕAψ_bpc = update(ϕAψ_bpc, PartitionEdge.(seq); message_update = ms -> default_message_update(ms; normalize = false))

    return ψ, ϕAψ_bpc
end


function bp_inserter(ψ::ITensorNetwork, ϕAψ_bpc::BeliefPropagationCache, v;
    cache_update_kwargs)

    local_state = normalize(dag(contract(environment(ϕAψ_bpc, [(v, "ket")]); sequence = "automatic")))
    ψ = copy(ψ)
    ψ[v] = copy(local_state)
    ϕAψ_bpc = update_factor(ϕAψ_bpc, (v, "ket"), local_state)
    ϕAψ = region_scalar(ϕAψ_bpc, only(partitionvertices(ϕAψ_bpc, [(v, "ket")])))

    return ψ, ϕAψ
end

function optimise(ϕ, A::ITensorNetwork, ψ::ITensorNetwork; cache_update_kwargs = (; maxiter = 20, tol = 1e-8),
    niters::Int64=10, overlap_tol = 1e-10)

    update_seq = default_update_seq(ψ)

    ϕAψs = zeros((niters, length(update_seq)))
    for i in 1:niters
        for (j, v) in enumerate(update_seq)
            ψ, ϕAψ_bpc = bp_extracter(ψ, A, ϕ, v; cache_update_kwargs)
            ψ, ϕAψs[i, j] = bp_inserter(ψ, ϕAψ_bpc, v; cache_update_kwargs)
        end


        if i >= 2 && abs(mean(ϕAψs[i,:]) - mean(ϕAψs[i-1,:])) / abs(mean(ϕAψs[i,:])) <= overlap_tol
            #println("Tolerance hit.")
            niters = i
            break
        end
    end

    #@show mean(ϕAψs[niters, :]), mean(ϕAψs[niters - 1, :])

    return norm_orthogonalize(ψ)
end

function optimise_two_site(ϕ, A::ITensorNetwork, ψ::ITensorNetwork; cache_update_kwargs = (; maxiter = 20, tol = 1e-8),
    niters::Int64=10, overlap_tol = 1e-10)

    ϕAψs = zeros((niters, length(update_seq)))
    for i in 1:niters
        for (j, v) in enumerate(update_seq)
            ψ, ϕAψ_bpc = bp_extracter(ψ, A, ϕ, v; cache_update_kwargs)
            ψ, ϕAψs[i, j] = bp_inserter(ψ, ϕAψ_bpc, v; cache_update_kwargs)
        end
        

        if i >= 2 && abs(mean(ϕAψs[i,:]) - mean(ϕAψs[i-1,:])) / abs(mean(ϕAψs[i,:])) <= overlap_tol
            #println("Tolerance hit.")
            niters = i
            break
        end
    end

    #@show mean(ϕAψs[niters, :]), mean(ϕAψs[niters - 1, :])

    return norm_orthogonalize(ψ)
end