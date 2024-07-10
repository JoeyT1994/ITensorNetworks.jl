using NamedGraphs.GraphsExtensions: vertices, src, dst, rem_edges
using NamedGraphs: NamedEdge
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs.PartitionedGraphs: PartitionEdge, partitionedges
using ITensors: ITensor, siteinds, contract, inds, prime, mapprime, dag, replaceinds, combiner, expect, commoninds
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
    L = 7
    g = named_grid((L,1); periodic = true)
    s = siteinds("S=1/2", g)
    ψ = random_tensornetwork(s; link_space = 3)
    cache_update_kwargs = (; maxiter = 10)
    ψIψ = QuadraticFormNetwork(ψ)
    ψIψ_bpc = BeliefPropagationCache(ψIψ)
    ψIψ_bpc = update(ψIψ_bpc; maxiter = 50)
    ψ, ψIψ_bpc = renormalize_update_norm_cache(ψ, ψIψ_bpc; cache_update_kwargs = (;), update_cache = false)
    v_focus = ((1,1))
    e = NamedEdge((4,1) => (5,1))


    ψIψ_bpc = normalize_messages(ψIψ_bpc)

    o_exact = only(expect(ψ, "Z", [v_focus]; alg = "exact"))
    println("Exact value is $o_exact")
    for i in 1:9
        O = effective_environment(ψ, ψIψ_bpc, v_focus, e, i; cache_update_kwargs)
        println("BP Value inserting $i dominant eigenvectors is $O")
        println("Absolute error is $(abs(O - o_exact))")
    end
end

main()