using ITensorNetworks: ITensorNetworks, AbstractITensorNetwork, BeliefPropagationCache, messages, partitioned_tensornetwork,
  optimal_contraction_sequence
using NamedGraphs.PartitionedGraphs: PartitionEdge, partitionedges, partitionvertices
using NamedGraphs.NamedGraphGenerators: named_hexagonal_lattice_graph
using NamedGraphs.GraphsExtensions: decorate_graph_edges
using NamedGraphs: rename_vertices
using ITensors: dag, replaceinds
using LinearAlgebra: norm, dot
using Dictionaries: Dictionary, set!
using OMEinsumContractionOrders

function normalize_messages(bp_cache::BeliefPropagationCache, pes::Vector{<:PartitionEdge})
  bp_cache = copy(bp_cache)
  mts = messages(bp_cache)
  for pe in pes
    me, mer = only(mts[pe]), only(mts[reverse(pe)])
    me /= norm(me)
    mer /= norm(mer)
    n = dot(me, mer)
    if isa(n, Float64) && n < 0 
      set!(mts, pe, ITensor[(-1 / sqrt(abs(n))) * me])
      set!(mts, reverse(pe), ITensor[(1 / sqrt(abs(n))) * mer])
    else
      set!(mts, pe, ITensor[(1 / sqrt(n)) * me])
      set!(mts, reverse(pe), ITensor[(1 / sqrt(n)) * mer])
    end
  end
  return bp_cache
end

function normalize_message(bp_cache::BeliefPropagationCache, pe::PartitionEdge)
  return normalize_messages(bp_cache, PartitionEdge[pe])
end

function normalize_messages(bp_cache::BeliefPropagationCache)
  return normalize_messages(bp_cache, partitionedges(partitioned_tensornetwork(bp_cache)))
end

function renamer(g)
    vertex_rename = Dictionary()
    for (i, v) in enumerate(vertices(g))
      set!(vertex_rename, v, (i,))
    end
    return rename_vertices(v -> vertex_rename[v], g)
end
  
function heavy_hex_lattice_graph(n::Int64, m::Int64; periodic)
    """Create heavy-hex lattice geometry"""
    g = named_hexagonal_lattice_graph(n, m; periodic)
    g = decorate_graph_edges(g)
    return renamer(g)
end

function lieb_lattice_graph(n::Int64, m::Int64; periodic)
  """Create heavy-hex lattice geometry"""
  g = named_grid((n,m); periodic)
  g = decorate_graph_edges(g)
  return renamer(g)
end

function renormalize_update_norm_cache(
  ψ::ITensorNetwork,
  ψIψ_bpc::BeliefPropagationCache;
  cache_update_kwargs,
  update_cache = true,
)

  ψ = copy(ψ)
  if update_cache
    ψIψ_bpc = update(ψIψ_bpc; cache_update_kwargs...)
  end
  ψIψ_bpc = normalize_messages(ψIψ_bpc)
  qf = tensornetwork(ψIψ_bpc)

  for v in vertices(ψ)
    v_ket, v_bra = ket_vertex(qf, v), bra_vertex(qf, v)
    pv = only(partitionvertices(ψIψ_bpc, [v_ket]))
    vn = region_scalar(ψIψ_bpc, pv)
    state = copy(ψ[v]) / sqrt(vn)
    state_dag = copy(dag(state))
    state_dag = replaceinds(
      state_dag, inds(state_dag), dual_index_map(qf).(inds(state_dag))
    )
    vertices_states = Dictionary([v_ket, v_bra], [state, state_dag])
    ψIψ_bpc = update_factors(ψIψ_bpc, vertices_states)
    ψ[v] = state
  end

  return ψ, ψIψ_bpc
end

function get_local_term(qf::QuadraticFormNetwork, v)
  return qf[ket_vertex(qf, v)] * qf[bra_vertex(qf, v)] * qf[operator_vertex(qf, v)]
end

function get_exact_environment(ψ::AbstractITensorNetwork, qf::QuadraticFormNetwork, v)
  ts = [get_local_term(qf, vp) for vp in setdiff(collect(vertices(ψ)), [v])]
  tn = ITensorNetwork(ts)
  seq = contraction_sequence(tn; alg = "sa_bipartite")
  return contract(tn; sequence = seq)
end

function ITensorNetworks.default_message_update(contract_list::Vector{ITensor}; normalize=true, kwargs...)
  sequence = optimal_contraction_sequence(contract_list)
  updated_messages = contract(contract_list; sequence, kwargs...)
  if normalize
    updated_messages /= norm(updated_messages)
  end
  return ITensor[updated_messages]
end