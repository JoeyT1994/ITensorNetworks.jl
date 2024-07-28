using NamedGraphs.GraphsExtensions: vertices, src, dst, rem_edges, eccentricity, vertices_at_distance
using NamedGraphs: NamedEdge
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs.PartitionedGraphs: PartitionEdge, partitionedges
using ITensors: ITensor, siteinds, contract, inds
using ITensorNetworks: ITensorNetwork, random_tensornetwork, QuadraticFormNetwork, bra_vertex, ket_vertex, operator_vertex, combine_linkinds, split_index, BeliefPropagationCache, update, message, partitioned_tensornetwork, messages, region_scalar,
    normalize_messages, tensornetwork, dual_index_map, update_factors, default_message, update_message, default_edge_sequence
using Graphs: center

using Random
include("contract_utils.jl")

function main()

    Random.seed!(2307)
    #Define the graph
    nx, ny = 5,5
    g = named_hexagonal_lattice_graph(nx, ny; periodic = false)
    #Define the index networks for state and op
    s = siteinds("S=1/2", g)
    state_chi = 2
    #Build random state, random op
    ψ = random_tensornetwork(s; link_space = state_chi)
    cache_update_kwargs = (; maxiter = 10)
    #Build the norm network <psi|A|psi>
    ψIψ = QuadraticFormNetwork(A, ψ)
    #Run BP on it
    ψIψ_bpc = BeliefPropagationCache(ψIψ)
    ψIψ_bpc = update(ψIψ_bpc; cache_update_kwargs...)
    #Get the centre of the lattice
    central_vert = first(center(ψ))
    dist = 3
    #Get the region containing all sites within distance 'dist' of the centre vert
    R = PartitionVertex.(unique(reduce(vcat, [vertices_at_distance(ψ, central_vert, d) for d in 0:dist])))
    println("Size of region is $(length(R))")
    #Take BP messages onto the boundary of R
    boundary_messages = environment(ψIψ_bpc, R)
    #Get the tensors in R from the norm network (except for the centre vert)
    central_tensors = reduce(vcat, [factor(ψIψ_bpc, pv) for pv in setdiff(R, [PartitionVertex(central_vert)])])
    #tn_R represents the tensor network which is an approximation to the environment of central_vert, accounting for
    #the region R exactly and the rest approximately.
    tn_R = ITensorNetwork(ITensor[boundary_messages; central_tensors])
    println("Size of network to contract is $(length(vertices(tn_R)))")
    seq = @time contraction_sequence(tn_R; alg="sa_bipartite")
    env = @time contract(tn_R; sequence=seq)
    
    result = contract([env; factor(ψIψ_bpc, PartitionVertex(central_vert))])
    @show result[]
end

main()