using Graphs: merge_vertices, has_edge, is_tree, center, degree
using NamedGraphs: NamedGraph, vertices, src, dst
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using NamedGraphs.GraphsExtensions: rem_vertex, subgraph, edges, neighbors, add_edges, add_edge, rem_edge,
    rem_edge!, rem_edges, post_order_dfs_edges, leaf_vertices, a_star, add_vertex, decorate_graph_edges, add_vertices!,
    add_edge!
using NamedGraphs: NamedEdge, rename_vertices
using NamedGraphs.PartitionedGraphs: PartitionEdge, partitionvertices, partitionedges, partitioned_graph,
    unpartitioned_graph, PartitionVertex, boundary_partitionedges, PartitionedGraph, partitioned_vertices,
    which_partition, partitionvertex, partitionedge
using ITensorNetworks: ITensorNetwork, AbstractITensorNetwork, IndsNetwork, random_tensornetwork, BeliefPropagationCache, QuadraticFormNetwork, ket_network,
    linkinds, underlying_graph, messages, indsnetwork, update, norm_sqr_network, message, factor, partitioned_tensornetwork, tensornetwork, normalize_messages,
    region_scalar, update_factors, default_edge_sequence, default_bp_maxiter, default_message_update, factors, ket_vertex, bra_vertex, contract_boundary_mps,
    dual_index_map
      combine_linkinds, neighbor_vertices
using ITensorNetworks.ModelHamiltonians: ising
using ITensors: ITensors, siteinds, delta, uniqueinds, Index, scalar, denseblocks, orthogonalize, inner, expect, apply,
  combiner, Trotter, commonind, replaceind, map_diag!
using ITensorMPS: ITensorMPS, MPS, MPO
using SplitApplyCombine: group
using Dictionaries: set!
using Random
using LinearAlgebra: LinearAlgebra, normalize, norm, tr
using Dictionaries: Dictionary

include("graph_utils.jl")
include("bp_utils.jl")
include("optimiser.jl")
include("boundary_mps_brute.jl")
include("beliefpropagationfunctions.jl")

function LinearAlgebra.normalize(
    ψ::ITensorNetwork,
    ψψ_bpc::BeliefPropagationCache;
    cache_update_kwargs,
    update_cache = true,
  )
    ψ = copy(ψ)
    if update_cache
      ψψ_bpc = update(ψψ_bpc; cache_update_kwargs...)
    end
    ψψ_bpc = normalize_messages(ψψ_bpc)
    ψψ = tensornetwork(ψψ_bpc)
  
    for v in vertices(ψ)
      v_ket, v_bra = (v, "ket"), (v, "bra")
      pv = only(partitionvertices(ψψ_bpc, [v_ket]))
      vn = region_scalar(ψψ_bpc, pv)
      state = copy(ψψ[v_ket]) / sqrt(vn)
      state_dag = copy(ψψ[v_bra]) / sqrt(vn)
      vertices_states = Dictionary([v_ket, v_bra], [state, state_dag])
      ψψ_bpc = update_factors(ψψ_bpc, vertices_states)
      ψ[v] = state
    end
  
    return ψ, ψψ_bpc
end

function norm_orthogonalize(ψ::AbstractITensorNetwork, v)
    ψ = orthogonalize(ψ, v)
    return ψ
end

function norm_orthogonalize(ψ::AbstractITensorNetwork, seq)
  ψ = orthogonalize(ψ, seq)
  return ψ
end

function MPS_truncate(ψ::AbstractITensorNetwork; cutoff = 1e-16)
    vs = sort(vertices(ψ))
    ψ_MPS = MPS(ψ)
    ψ_MPS = truncate(ψ_MPS; cutoff)
    return ITensorNetwork([v => ψ_MPS[i] for (i,v) in enumerate(vs)])
end

function flatten_quadratic_form(ψOψ::QuadraticFormNetwork)
  return ITensorNetwork([v => ψOψ[(v, "ket")]*ψOψ[(v, "operator")]*ψOψ[(v, "bra")] for v in unique(first.(vertices(ψOψ)))])
end

function combine_siteinds(ψ::ITensorNetwork)
  ψ = copy(ψ)
  s = siteinds(ψ)
  for v in vertices(ψ)
    ψ[v] = ψ[v] * combiner(s[v])
  end
  return ψ
end