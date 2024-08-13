using ITensors: ITensors, scalartype, which_op, name, names, sites, OpSum
using ITensorNetworks:
  AbstractITensorNetwork, ITensorNetwork, BeliefPropagationCache, ket_vertices, bra_vertices, tensornetwork, default_message_update, operator_network
using ITensorNetworks.ITensorsExtensions: map_eigvals
using NamedGraphs.GraphsExtensions: a_star, neighbors, boundary_edges, vertices_at_distance
using OMEinsumContractionOrders

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
    e_region = vcat([bra_vertex(ψOψ_qf, v) for v in region], [ket_vertex(ψOψ_qf, v) for v in region])
    push!(o_environments, environment(ψOψ_bpc, e_region))      
  end
  return o_environments, n_environment
end

function effective_environments_enlarged_region(
  state::ITensorNetwork, H::OpSum, ψIψ_bpc::BeliefPropagationCache, region, central_vert
)
  s = indsnetwork(state)
  @assert central_vert ∈ region

  operators = get_tnos(s, H, [central_vert])
  op_environments = Vector{ITensor}[]
  ψIψ_qf = tensornetwork(ψIψ_bpc)
  e_region = vcat(
    [bra_vertex(ψIψ_qf, v) for v in region], [ket_vertex(ψIψ_qf, v) for v in region]
  )
  norm_central_state_tensors = vcat(
    ITensor[ψIψ_qf[ket_vertex(ψIψ_qf, v)] for v in setdiff(region, [central_vert])],
    ITensor[ψIψ_qf[bra_vertex(ψIψ_qf, v)] for v in setdiff(region, [central_vert])],
  )
  norm_environments = vcat(environment(ψIψ_bpc, e_region), norm_central_state_tensors)
  for operator in operators
    ψOψ_qf = QuadraticFormNetwork(operator, state)
    ψOψ_bpc = BeliefPropagationCache(ψOψ_qf)
    broken_edges = setdiff(edges(state), edges(operator))
    mts = messages(ψOψ_bpc)
    for be in edges(state)
      set!(mts, PartitionEdge(be), message(ψIψ_bpc, PartitionEdge(be)))
      set!(mts, PartitionEdge(reverse(be)), message(ψIψ_bpc, PartitionEdge(reverse(be))))
    end

    partition_edge_sequence =
      PartitionEdge.(post_order_dfs_edges(underlying_graph(operator), region))
    ψOψ_bpc = update(
      ψOψ_bpc,
      partition_edge_sequence;
      message_update=ms -> default_message_update(ms; normalize=false),
    )
    op_central_state_tensors = vcat(
      ITensor[ψOψ_qf[ket_vertex(ψOψ_qf, v)] for v in setdiff(region, [central_vert])],
      ITensor[ψOψ_qf[bra_vertex(ψOψ_qf, v)] for v in setdiff(region, [central_vert])],
    ) 
    push!(op_environments, vcat(environment(ψOψ_bpc, e_region), op_central_state_tensors))
  end
  return op_environments, norm_environments
end


function bp_extracter(
  ψ::AbstractITensorNetwork,
  H::OpSum,
  ψIψ_bpc::BeliefPropagationCache,
  region;
  dist::Int = 0
)
  state = prod([ψ[v] for v in region])
  if dist == 0
    ∂ψOψ_bpc_∂rs, ∂ψIψ_bpc_∂r = effective_environments(ψ, H, ψIψ_bpc, region)
  else
    central_vert = only(region)
    super_region = unique(reduce(vcat,[vertices_at_distance(ψ, central_vert, d) for d in 0:dist]))
    ∂ψOψ_bpc_∂rs, ∂ψIψ_bpc_∂r = effective_environments_enlarged_region(ψ, H, ψIψ_bpc, super_region, central_vert)
  end

  return state, ∂ψOψ_bpc_∂rs, ∂ψIψ_bpc_∂r
end