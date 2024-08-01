using ITensorNetworks: AbstractITensorNetwork, BeliefPropagationCache, messages, partitioned_tensornetwork
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
    ψOψ_bpc = update(ψOψ_bpc, partition_edge_sequence)
    e_region = vcat([bra_vertex(ψOψ_qf, v) for v in region], [ket_vertex(ψOψ_qf, v) for v in region])
    push!(environments, environment(ψOψ_bpc, e_region))      
  end
  return environments
end