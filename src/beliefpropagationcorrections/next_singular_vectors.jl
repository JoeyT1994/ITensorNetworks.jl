using NamedGraphs.GraphsExtensions: vertices, src, dst, rem_edges
using NamedGraphs: NamedEdge
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs.PartitionedGraphs: PartitionEdge, partitionedges
using ITensors: ITensor, siteinds, contract, inds, prime, mapprime, dag, replaceinds, combiner
using ITensors.NDTensors: svd, eigen, array
using ITensorNetworks: ITensorNetwork, random_tensornetwork, QuadraticFormNetwork, bra_vertex, ket_vertex, operator_vertex, combine_linkinds, split_index, BeliefPropagationCache, update, message, partitioned_tensornetwork, messages, region_scalar,
    normalize_messages, tensornetwork, dual_index_map, update_factors, default_message, update_message, default_edge_sequence
using LinearAlgebra: norm, eigvals, dot, diag
using Dictionaries: Dictionary, set!
using KrylovKit: eigsolve

using Random

function project(ψIψ_bpc::BeliefPropagationCache, pe::PartitionEdge, m_pes::Vector{ITensor}, m_pers::Vector{ITensor})
    ψIψ_bpc = copy(ψIψ_bpc)
    ms = messages(ψIψ_bpc)
    per = reverse(pe)
    me, mer = only(message(ψIψ_bpc, pe)), only(message(ψIψ_bpc, per))
    for (ml, mr) in zip(m_pes, m_pers)
        me = me - me * (mr) * ml
        mer = mer - mer * (ml) * mr
    end
    me /= norm(me)
    mer /= norm(mer)
    set!(ms, pe, ITensor[me])
    set!(ms, per, ITensor[mer])
    return ψIψ_bpc
end 

#Given an updated cache get the next message
function get_eigenvectors(ψIψ_bpc::BeliefPropagationCache, pe::PartitionEdge, howmany::Int64 = 1; maxiter::Int64 = 20)
    per = reverse(pe)
    left_vectors, right_vectors = ITensor[copy(only(message(ψIψ_bpc, per)))], ITensor[copy(only(message(ψIψ_bpc, pe)))]
    #Going parallel seems crucial?!
    pes = [[pe] for pe in default_edge_sequence(ψIψ_bpc)]
    for k in 2:howmany
        ψIψ_bpc = BeliefPropagationCache(partitioned_tensornetwork(ψIψ_bpc))
        ψIψ_bpc = project(ψIψ_bpc, pe, right_vectors, left_vectors)
        for i in 1:maxiter
            ψIψ_bpc = update(ψIψ_bpc, pes)
            ψIψ_bpc = project(ψIψ_bpc, pe, right_vectors, left_vectors)
        end
        n = region_scalar(ψIψ_bpc, pe)
        if n > 0 
            push!(right_vectors, (1 / sqrt(n))*only(message(ψIψ_bpc, pe)))
            push!(left_vectors, (1 / sqrt(n))*only(message(ψIψ_bpc, per)))
        elseif n < 0
            push!(right_vectors, (-1 / sqrt(abs(n)))*only(message(ψIψ_bpc, pe)))
            push!(left_vectors, (1 / sqrt(abs(n)))*only(message(ψIψ_bpc, per)))
        else
            push!(right_vectors, only(message(ψIψ_bpc, pe)))
            push!(left_vectors, only(message(ψIψ_bpc, per)))
        end
    end

    return left_vectors, right_vectors
end

Random.seed!(1437)
L = 4
g = named_grid((L,L); periodic = true)
s = siteinds("S=1/2", g)
ψ = random_tensornetwork(s; link_space = 2)
ψIψ = QuadraticFormNetwork(ψ)
ψIψ_bpc = BeliefPropagationCache(ψIψ)
ψIψ_bpc = update(ψIψ_bpc; cache_update_kwargs...)
ψIψ_bpc = normalize_messages(ψIψ_bpc)

vsrc, vdst = (1,1), (2,1)
e = NamedEdge(vsrc => vdst)
pe = PartitionEdge(e)
howmany = 4
left_vectors, right_vectors = get_eigenvectors(ψIψ_bpc, pe, howmany; maxiter = 30)

@show [dot(left_vectors[i], right_vectors[j]) for i in 1:howmany for j in 1:howmany]
Cr, Cl = combiner(inds(first(right_vectors))), combiner(inds(first(left_vectors)))

@show sum([(left_vectors[i]*Cl)*(right_vectors[i]*Cr) for i in 1:howmany])