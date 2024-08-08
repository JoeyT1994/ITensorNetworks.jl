using ITensors: scalartype, which_op, name, names, sites, OpSum
using ITensorNetworks:
  AbstractITensorNetwork, ket_vertices, bra_vertices, tensornetwork, default_message_update, operator_network
using ITensorNetworks.ITensorsExtensions: map_eigvals
using NamedGraphs.GraphsExtensions: a_star, neighbors, boundary_edges

include("treetensornetworkoperators.jl")

function effective_environments(state::ITensorNetwork, H::OpSum, ψIψ_bpc::BeliefPropagationCache, region)
  s = indsnetwork(state)

  operators = get_tnos(s, H, region)
  o_environments = Vector{ITensor}[]
  ψIψ_qf = tensornetwork(ψIψ_bpc)
  n_region = vcat([bra_vertex(ψIψ_qf, v) for v in region], [ket_vertex(ψIψ_qf, v) for v in region])
  n_environment = environment(ψIψ_bpc, n_region)
  for operator in operators
    ψOψ_qf = QuadraticFormNetwork(operator, state)
    ψOψ_bpc = BeliefPropagationCache(ψOψ_qf)
    broken_edges = setdiff(edges(state), edges(operator))
    mts = messages(ψOψ_bpc)
    for be in broken_edges
      set!(mts, PartitionEdge(be), message(ψIψ_bpc, PartitionEdge(be)))
      set!(mts, PartitionEdge(reverse(be)), message(ψIψ_bpc, PartitionEdge(reverse(be))))
    end

    partition_edge_sequence = PartitionEdge.(post_order_dfs_edges(underlying_graph(operator), first(region)))
    partition_edge_sequence = filter(e -> src(e) ∉ PartitionVertex.(region), partition_edge_sequence)
    ψOψ_bpc = update(ψOψ_bpc, partition_edge_sequence; message_update = mts -> default_message_update(mts; normalize = false))
    ψIψ_bpc = update(ψIψ_bpc, partition_edge_sequence; message_update = mts -> default_message_update(mts; normalize = false))
    e_region = vcat([bra_vertex(ψOψ_qf, v) for v in region], [ket_vertex(ψOψ_qf, v) for v in region])
    push!(o_environments, environment(ψOψ_bpc, e_region))      
  end
  return o_environments, n_environment
end

function bp_extracter(
  ψ::AbstractITensorNetwork,
  H::OpSum,
  ψIψ_bpc::BeliefPropagationCache,
  region;
)

  form_network = tensornetwork(ψIψ_bpc)
  form_ket_vertices, form_bra_vertices = ket_vertices(form_network, region),
  bra_vertices(form_network, region)

  ∂ψOψ_bpc_∂rs, ∂ψIψ_bpc_∂r = effective_environments(ψ, H, ψIψ_bpc, region)
  state = prod([ψ[v] for v in region])

  return state, ∂ψOψ_bpc_∂rs, ∂ψIψ_bpc_∂r
end