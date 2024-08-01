using NamedGraphs.GraphsExtensions: vertices, src, dst, rem_edges, eccentricity, vertices_at_distance
using NamedGraphs: NamedEdge, nv
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

    Random.seed!(2807)
    #Define the graph
    nx, ny = 5,5
    g = lieb_lattice_graph(nx, ny; periodic = false)
    #g = named_grid((nx,ny))
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
    ψIψ = tensornetwork(ψIψ_bpc)

    #Get the centre of the lattice
    central_vert = first(center(ψ))

    H_opsum = ising(g; h = 0.5)
    envs = effective_environments_enlarged_region(ψ, H_opsum, ψIψ_bpc,[central_vert], central_vert )
    bp_energy = sum([contract([env; ψIψ[ket_vertex(ψIψ, central_vert)]; ψIψ[bra_vertex(ψIψ, central_vert)]]; sequence = "automatic")[] for env in envs])
    println("BP energy is $bp_energy")

    #Get the region containing all sites within distance 'dist' of the centre vert
    dist = 6
    R = unique(reduce(vcat, [vertices_at_distance(ψ, central_vert, d) for d in 0:dist]))

    println("Size of region is $(length(R))")
    op_envs = effective_environments_enlarged_region(ψ, H_opsum, ψIψ_bpc,R, central_vert )
    op_vectorized_lists = [ITensor[env; ψIψ[ket_vertex(ψIψ, central_vert)]; ψIψ[bra_vertex(ψIψ, central_vert)]] for env in op_envs]
    op_seqs = [contraction_sequence(vectorized_list; alg="sa_bipartite") for vectorized_list in op_vectorized_lists]
    numerator = sum([contract(vectorized_list; sequence = seq)[] for (seq, vectorized_list) in zip(op_seqs, op_vectorized_lists)])
    norm_env = effective_norm_environments_enlarged_region(ψ, ψIψ_bpc,R, central_vert )
    norm_vectorized_list = ITensor[norm_env; ψIψ[ket_vertex(ψIψ, central_vert)]; ψIψ[bra_vertex(ψIψ, central_vert)]]
    norm_seq = contraction_sequence(norm_vectorized_list; alg="sa_bipartite")
    denominator = contract(norm_vectorized_list; sequence = norm_seq)[]
    bp_energy_corrected = numerator / denominator
    println("Corrected BP energy is $bp_energy_corrected")

    tnos = get_tnos(s, H_opsum, [central_vert])
    ψAψs = QuadraticFormNetwork.(tnos, (ψ,))
    seqs = contraction_sequence.(ψAψs; alg="sa_bipartite")
    norm_seq = contraction_sequence(ψIψ; alg = "sa_bipartite")
    exact_energy = sum([contract(ψAψ; sequence = seq)[] for (ψAψ, seq) in zip(ψAψs, seqs)]) / contract(ψIψ; sequence = norm_seq)[]
    println("Exact energy is $exact_energy")

    
end

main()