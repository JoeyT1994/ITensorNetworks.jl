function build_bp_cache(ψ::AbstractITensorNetwork; kwargs...)
    bpc = BeliefPropagationCache(QuadraticFormNetwork(ψ))
    bpc = update(bpc; kwargs...)
    return bpc
end

function ITensors.apply(
    o::ITensor,
    ψ::AbstractITensorNetwork,
    bpc::BeliefPropagationCache;
    reset_all_messages = false,
    apply_kwargs...,
)
    bpc = copy(bpc)
    vs = neighbor_vertices(ψ, o)
    envs = environment(bpc, PartitionVertex.(vs))
    singular_values! = Ref(ITensor())
    ψ = noprime(apply(o, ψ; envs, singular_values!, apply_kwargs...))
    if length(vs) == 2
        v1, v2 = vs
        pe = partitionedge(bpc, (v1, "bra") => (v2, "bra"))
        mts = messages(bpc)
        ind2 = commonind(singular_values![], ψ[v1])
        δuv = dag(copy(singular_values![]))
        δuv = replaceind(δuv, ind2, ind2')
        map_diag!(sign, δuv, δuv)
        singular_values![] = denseblocks(singular_values![]) * denseblocks(δuv)
        if !reset_all_messages
            set!(mts, pe, dag.(ITensor[singular_values![]]))
            set!(mts, reverse(pe), ITensor[singular_values![]])
        else
            bpc = BeliefPropagationCache(partitioned_tensornetwork(bpc))
        end
    end
    for v in vs
        ψdag_v = dual_index_map(tensornetwork(bpc))(dag(ψ[v]))
        vertices_factors = Dict(zip([(v, "ket"), (v, "bra")], [ψ[v], ψdag_v]))
        bpc = update_factors(bpc, vertices_factors)
    end
    return ψ, bpc
end

#Note that region should consist of contiguous vertices here!
function rdm(ψ::ITensorNetwork, region; (cache!) = nothing, cache_update_kwargs = (;))
    cache = isnothing(cache!) ? build_bp_cache(ψ; cache_update_kwargs...) : cache![]
    ψIψ = tensornetwork(cache)

    state_tensors = vcat(
        ITensor[ψIψ[ket_vertex(ψIψ, v)] for v in region],
        ITensor[ψIψ[bra_vertex(ψIψ, v)] for v in region],
    )
    env = environment(cache, PartitionVertex.(region))

    rdm = contract(ITensor[env; state_tensors]; sequence = "automatic")

    rdm = array((rdm * combiner(inds(rdm; plev = 0)...)) * combiner(inds(rdm; plev = 1)...))
    rdm /= tr(rdm)

    return rdm
end

function two_site_expect(ψIψ::BeliefPropagationCache, v1, v2, op1::String, op2::String)
    ψIψ_qf = tensornetwork(ψIψ)
    denominator = path_contract(ψIψ, v1, v2)
    ov1, ov2 = operator_vertex(ψIψ_qf, v1), operator_vertex(ψIψ_qf, v2)
    s1, s2 = commonind(ψIψ_qf[ket_vertex(ψIψ_qf, v1)], ψIψ_qf[ov1]),
    commonind(ψIψ_qf[ket_vertex(ψIψ_qf, v2)], ψIψ_qf[ov2])
    o1, o2 = ITensors.op(op1, s1), ITensors.op(op2, s2)

    ψOψ = update_factor(ψIψ, ov1, o1)
    ψOψ = update_factor(ψOψ, ov2, o2)
    numerator = path_contract(ψOψ, v1, v2)

    return numerator / denominator
end

function path_contract(ψAψ::BeliefPropagationCache, v1, v2)
    pg = partitioned_tensornetwork(ψAψ)
    path = PartitionEdge.(a_star(partitioned_graph(ψAψ), v1, v2))
    ψAψ = update(
        ψAψ,
        path;
        message_update = ms -> default_message_update(ms; normalize = false),
    )
    return region_scalar(ψAψ, PartitionVertex(v2))
end
