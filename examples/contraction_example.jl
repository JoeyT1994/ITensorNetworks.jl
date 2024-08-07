using NamedGraphs.GraphsExtensions:
  vertices, src, dst, rem_edges, eccentricity, vertices_at_distance
using NamedGraphs: NamedEdge, nv
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs.PartitionedGraphs: PartitionEdge, partitionedges, PartitionVertex
using ITensors: ITensors, ITensor, siteinds, contract, inds, commonind
using ITensorNetworks:
  ITensorNetwork,
  random_tensornetwork,
  QuadraticFormNetwork,
  bra_vertex,
  ket_vertex,
  operator_vertex,
  combine_linkinds,
  split_index,
  BeliefPropagationCache,
  update,
  message,
  partitioned_tensornetwork,
  messages,
  region_scalar,
  tensornetwork,
  dual_index_map,
  update_factors,
  default_message,
  update_message,
  default_edge_sequence,
  environment,
  factor,
  contraction_sequence
using Graphs: center
using OMEinsumContractionOrders: OMEinsumContractionOrders
using ITensorNetworks.ModelHamiltonians: ising, heisenberg

using Random
include("contract_utils.jl")
include("tree_tensornetwork_operators.jl")

ITensors.disable_warn_order()

function main()
  Random.seed!(2807)
  #Define the graph
  nx, ny = 8, 8
  g = lieb_lattice_graph(nx, ny; periodic=true)
  #Define the index networks for state and op
  s = siteinds("S=1/2", g)
  #Define the bond dimension of the state
  state_chi = 3
  #Define the Hamiltonian
  H = heisenberg
  #Build the state, random or one generated via imaginary tine evo
  ψ = random_tensornetwork(s; link_space=state_chi)
  #ψ = random_tensornetwork(s; link_space = 1)
  #ψ = imaginary_time_evo(s, ψ, H, [(10,0.5), (10, 0.25), (10, 0.1), (10, 0.01)]; model_params = (; ), apply_kwargs = (; maxdim = state_chi, cutoff = 1e-12))

  cache_update_kwargs = (; maxiter=10)
  #Build the norm network <psi|psi>
  ψIψ = QuadraticFormNetwork(ψ)
  #Run BP on it and renormalize it so the BP norm is 1
  ψIψ_bpc = BeliefPropagationCache(ψIψ)
  ψ, ψIψ_bpc = renormalize_update_norm_cache(ψ, ψIψ_bpc; cache_update_kwargs)
  ψIψ = tensornetwork(ψIψ_bpc)

  #Get the centre of the lattice
  central_vert = first(center(ψ))

  #Define the list of operators in the Hamiltonian
  H_opsum = H(g)

  #Define the sizes of the super regions to contract
  dists = [0, 4, 6]

  #Extract the tensors at the centre of the lattice
  state_tensors = ITensor[
    ψIψ[ket_vertex(ψIψ, central_vert)], ψIψ[bra_vertex(ψIψ, central_vert)]
  ]

  for dist in dists
    @time begin
      #Construct the super region R
      R = unique(reduce(vcat, [vertices_at_distance(ψ, central_vert, d) for d in 0:dist]))
      println("Distance is $dist")
      println("Size of region is $(length(R))")

      #Pull out the tensor network which corresponds to the environment for the centre site
      op_envs, norm_env = effective_environments_enlarged_region(
        ψ, H_opsum, ψIψ_bpc, R, central_vert
      )

      #Get conraction seqs
      op_seqs = [contraction_sequence(op_env; alg="sa_bipartite") for op_env in op_envs]
      norm_seq = contraction_sequence(norm_env; alg="sa_bipartite")

      #Contract the environment
      op_envs = [contract(op_env; sequence=seq) for (seq, op_env) in zip(op_seqs, op_envs)]
      norm_env = contract(norm_env; sequence=norm_seq)

      #Contract the environment with the local state to get an approximate energy, as |R| -> lattice diameter this becomes exact
      denominator = contract(ITensor[norm_env; state_tensors]; sequence="automatic")[]
      numerators = [
        contract(ITensor[op_env; state_tensors]; sequence="automatic")[] for
        op_env in op_envs
      ]

      bp_energy_corrected = sum(numerators) / denominator
      println("Corrected BP energy is $bp_energy_corrected")
    end
  end
end

main()
