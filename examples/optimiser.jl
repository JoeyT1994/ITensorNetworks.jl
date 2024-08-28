using ITensorNetworks: BilinearFormNetwork, norm_sqr_network, BeliefPropagationCache, environment, update_factor,
    disjoint_union
using ITensors: ITensor, inds, contract, dag, prime, replaceinds, commoninds, noprime
using ITensorNetworks.ITensorsExtensions: map_eigvals
using ITensorNetworks: contraction_sequence
using OMEinsumContractionOrders
using Statistics

function default_update_seq(ψ::AbstractITensorNetwork; nsites::Int = 1)
    vs = leaf_vertices(ψ)
    @assert length(vs) == 2
    vstart, vend = first(vs), last(vs)
    path = a_star(ψ, vstart, vend)
    return vcat(src.(path), dst.(reverse(path)))
end

function bp_extracter(ϕAψ_bpc::BeliefPropagationCache, v)
    return normalize(contract(environment(ϕAψ_bpc, [(v, "ket")]); sequence = "automatic"))
end

function bp_inserter(local_state::ITensor, ψ::ITensorNetwork, ϕAψ_bpc::BeliefPropagationCache, v)
    ψ = copy(ψ)
    ψ[v] = dag(local_state)
    ϕAψ_bpc = update_factor(ϕAψ_bpc, (v, "ket"), dag(local_state))
    ϕAψ = region_scalar(ϕAψ_bpc, only(partitionvertices(ϕAψ_bpc, [(v, "ket")])))
    return ψ, ϕAψ_bpc, ϕAψ
end

function bp_updater(ψ::ITensorNetwork, ϕAψ_bpc::BeliefPropagationCache, v, cur_ortho_centre)
    if cur_ortho_centre != nothing
        seq = a_star(ψ, cur_ortho_centre, v)
        p_seq = a_star(partitioned_graph(ϕAψ_bpc), parent(only(partitionvertices(ϕAψ_bpc, [(cur_ortho_centre, "ket")]))), parent(only(partitionvertices(ϕAψ_bpc, [(v, "ket")]))))
        ψ = orthogonalize(ψ, seq)
        vertices_factors = Dictionary([(v, "ket") for v in vcat(src.(seq), [v])], [ψ[v] for v in vcat(src.(seq), [v])])
    else
        seq = v
        p_seq = post_order_dfs_edges(partitioned_graph(ϕAψ_bpc), parent(only(partitionvertices(ϕAψ_bpc, [(v, "ket")]))))
        ψ = orthogonalize(ψ, v)
        vertices_factors = Dictionary([(v, "ket") for v in collect(vertices(ψ))], [ψ[v] for v in vertices(ψ)])
    end
    ϕAψ_bpc = update_factors(ϕAψ_bpc, vertices_factors)
    ϕAψ_bpc = update(ϕAψ_bpc, PartitionEdge.(p_seq); message_update = ms -> default_message_update(ms; normalize = false))
    return ψ, ϕAψ_bpc, v
end


function optimise(ϕ, A::ITensorNetwork, ψ::ITensorNetwork; cache_update_kwargs = (; maxiter = 20, tol = 1e-8),
    niters::Int64=10, overlap_tol = 1e-10)

    update_seq = default_update_seq(ψ)
    ϕAψ_bpc =build_sandwich(ψ, A, ϕ)
    ortho_centre = nothing
    ϕAψs = zeros(ComplexF64, (niters, length(update_seq)))
    for i in 1:niters
        for (j, v) in enumerate(update_seq)
            ψ, ϕAψ_bpc, ortho_centre = bp_updater(ψ, ϕAψ_bpc, v, ortho_centre)
            local_state = bp_extracter(ϕAψ_bpc, v)
            ψ, ϕAψ_bpc, ϕAψs[i, j] = bp_inserter(local_state, ψ, ϕAψ_bpc, v)
            #@show ϕAψs[i, j]
        end


        if i >= 2 && abs(mean(ϕAψs[i,:]) - mean(ϕAψs[i-1,:])) / abs(mean(ϕAψs[i,:])) <= overlap_tol
            #println("Tolerance hit.")
            niters = i
            break
        end
    end
    #@show mean(ϕAψs[niters, :]), mean(ϕAψs[niters - 1, :])
    return dag(ψ)
end