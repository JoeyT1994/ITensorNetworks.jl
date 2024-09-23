using ITensors: scalartype, which_op, name, names, sites, OpSum
using ITensorNetworks:
  AbstractITensorNetwork, ket_vertices, bra_vertices, tensornetwork, default_message_update, operator_network
using ITensorNetworks.ITensorsExtensions: map_eigvals
using NamedGraphs.GraphsExtensions: a_star, neighbors, boundary_edges

include("treetensornetworkoperators.jl")

function effective_environments(state::ITensorNetwork, H::OpSum, ψIψ_bpc::BeliefPropagationCache, region)
  s = indsnetwork(state)

  operators = get_tnos(s, H, first(region))
  environments = Vector{ITensor}[]
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
    push!(environments, environment(ψOψ_bpc, e_region))      
  end
  return environments
end

function bp_extracter(
  ψ::AbstractITensorNetwork,
  H::OpSum,
  ψIψ_bpc::BeliefPropagationCache,
  region;
  regularization=10 * eps(scalartype(ψ)),
  ishermitian=true,
)

  form_network = tensornetwork(ψIψ_bpc)
  form_ket_vertices, form_bra_vertices = ket_vertices(form_network, region),
  bra_vertices(form_network, region)

  ∂ψOψ_bpc_∂rs = effective_environments(ψ, H, ψIψ_bpc, region)
  state = prod([ψ[v] for v in region])
  messages = environment(ψIψ_bpc, partitionvertices(ψIψ_bpc, form_ket_vertices))
  f_sqrt = sqrt ∘ (x -> x + regularization)
  f_inv_sqrt = inv ∘ sqrt ∘ (x -> x + regularization)
  sqrt_mts =
    map_eigvals.(
      (f_sqrt,), messages, first.(inds.(messages)), last.(inds.(messages)); ishermitian
    )
  inv_sqrt_mts =
    map_eigvals.(
      (f_inv_sqrt,), messages, first.(inds.(messages)), last.(inds.(messages)); ishermitian
    )

  return state, ∂ψOψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts
end