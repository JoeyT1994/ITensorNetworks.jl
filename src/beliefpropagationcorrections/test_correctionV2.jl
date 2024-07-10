using Graphs: SimpleGraph, cycle_basis
using NamedGraphs.GraphsExtensions: vertices, src, dst, rem_edges
using NamedGraphs: NamedEdge
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs.PartitionedGraphs: PartitionEdge, partitionedges, PartitionVertex
using ITensors: ITensor, siteinds, contract, inds, prime, mapprime, dag, replaceinds, combiner, expect, commoninds, permute
using ITensors.NDTensors: svd, eigen, array
using ITensorNetworks: ITensorNetwork, random_tensornetwork, QuadraticFormNetwork, bra_vertex, ket_vertex, operator_vertex, combine_linkinds, split_index, BeliefPropagationCache, update, message, partitioned_tensornetwork, messages, region_scalar,
    normalize_messages, tensornetwork, dual_index_map, update_factors, default_message, update_message, default_edge_sequence
using LinearAlgebra: norm, eigvals, dot, diag
using Dictionaries: Dictionary, set!
using KrylovKit: eigsolve

using Random

include("utils.jl")
include("../beliefpropagationdmrg/bp_inserter.jl")

function main()

    Random.seed!(2307)
    L = 9
    g = named_grid((L,1); periodic = true)
    s = siteinds("S=1/2", g)
    ψ = random_tensornetwork(s; link_space = 3)
    cache_update_kwargs = (; maxiter = 10)
    ψIψ = QuadraticFormNetwork(ψ)
    ψIψ_bpc = BeliefPropagationCache(ψIψ)
    ψIψ_bpc = update(ψIψ_bpc; maxiter = 50)
    ψ, ψIψ_bpc = renormalize_update_norm_cache(ψ, ψIψ_bpc; cache_update_kwargs = (;), update_cache = false)
    v_focus = (1,1)
    o_str = "Z"
    oper = ITensors.op(o_str, s[v_focus])
    ket_factor, bra_factor = factor(ψIψ_bpc, ket_vertex(ψIψ, v_focus)), factor(ψIψ_bpc, bra_vertex(ψIψ, v_focus))
    id_op = factor(ψIψ_bpc, operator_vertex(ψIψ, v_focus))


    #Random SVD instead?!
    howmany = 4
    r_vecs, l_vecs, r_vals, l_vals = eigendecompose_loop(ψ, ψIψ_bpc, v_focus, howmany)

    for no_eigs in 1:length(r_vals)
        numerator = sum(contract([r_vals[i]*r_vecs[i], l_vecs[i], ket_factor, bra_factor, oper]; sequence = "automatic") for i in 1:no_eigs)[]
        denominator = sum(contract([r_vals[i]*r_vecs[i], l_vecs[i], ket_factor, bra_factor, id_op]; sequence = "automatic") for i in 1:no_eigs)[]
        @show numerator / denominator
    end

end

main()