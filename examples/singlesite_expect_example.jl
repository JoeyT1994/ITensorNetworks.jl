using NamedGraphs.GraphsExtensions: vertices, src, dst, rem_edges, eccentricity, vertices_at_distance
using NamedGraphs: NamedEdge
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs.PartitionedGraphs: PartitionEdge, partitionedges, PartitionVertex
using ITensors: ITensors, ITensor, siteinds, contract, inds, commonind
using ITensorNetworks: ITensorNetwork, random_tensornetwork, QuadraticFormNetwork, bra_vertex, ket_vertex, operator_vertex, combine_linkinds, split_index, BeliefPropagationCache, update, message, partitioned_tensornetwork, messages, region_scalar,
    tensornetwork, dual_index_map, update_factors, default_message, update_message, default_edge_sequence, environment, factor, contraction_sequence
using Graphs: center
using OMEinsumContractionOrders: OMEinsumContractionOrders
using ITensorNetworks.ModelHamiltonians: ising

using Random
include("contract_utils.jl")
include("tree_tensornetwork_operators.jl")

ITensors.disable_warn_order()

function main()

    Random.seed!(2307)
    #Define the graph
    nx, ny = 5,5
    g = lieb_lattice_graph(nx, ny; periodic = true)
    #Define the index networks for state and op
    s = siteinds("S=1/2", g)
    state_chi = 2
    #Build random state, random op
    ψ = random_tensornetwork(s; link_space = state_chi)
    cache_update_kwargs = (; maxiter = 10)
    #Build the norm network <psi|A|psi>
    ψIψ = QuadraticFormNetwork(ψ)
    #Run BP on it
    ψIψ_bpc = BeliefPropagationCache(ψIψ)
    ψ, ψIψ_bpc = renormalize_update_norm_cache(ψ, ψIψ_bpc; cache_update_kwargs)

    #Get the centre of the lattice
    central_vert = first(center(ψ))
    dist = 6
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
    env_exact = get_exact_environment(ψ, ψIψ, central_vert)
    
    ψIψ_v = ψIψ[operator_vertex(ψIψ, central_vert)]
    s = commonind(ψIψ[ket_vertex(ψIψ, central_vert)], ψIψ_v)
    operator = ITensors.op("Z", s)
    local_numerator_state = [operator, ψIψ[ket_vertex(ψIψ, central_vert)], ψIψ[bra_vertex(ψIψ, central_vert)]]
    local_denominator_state = [ψIψ_v, ψIψ[ket_vertex(ψIψ, central_vert)], ψIψ[bra_vertex(ψIψ, central_vert)]]

    num_approx = contract([env; local_numerator_state])
    denom_approx = contract([env; local_denominator_state])
    z_approx = num_approx[] / denom_approx[]
    
    num_exact = contract([env_exact; local_numerator_state])
    denom_exact = contract([env_exact; local_denominator_state])
    z_exact = num_exact[] / denom_exact[]

    @show z_approx, z_exact
end

main()