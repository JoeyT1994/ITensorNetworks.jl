using Graphs: merge_vertices
using NamedGraphs: vertices, src, dst
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs.GraphsExtensions: rem_vertex, subgraph, edges, neighbors
using NamedGraphs: NamedEdge, rename_vertices
using NamedGraphs.PartitionedGraphs: PartitionEdge, partitionvertices, partitionedges, partitioned_graph,
    unpartitioned_graph, PartitionVertex, boundary_partitionedges
using ITensorNetworks: ITensorNetwork, AbstractITensorNetwork, IndsNetwork, random_tensornetwork, BeliefPropagationCache, QuadraticFormNetwork, ket_network,
    linkinds, underlying_graph, messages, indsnetwork, update, norm_sqr_network, message, factor, partitioned_tensornetwork, tensornetwork, normalize_messages,
    region_scalar, update_factors
using ITensors: siteinds, delta, uniqueinds, Index, scalar, denseblocks
using SplitApplyCombine: group
using Dictionaries: set!
using Random
using LinearAlgebra: LinearAlgebra, normalize
using Dictionaries: Dictionary

include("bp_utils.jl")

function named_grid_periodic_x(nx::Int64, ny::Int64)
    g = named_grid((nx,ny))
    for i in 1:ny
        g = add_edge(g, NamedEdge((i, nx) => (i, 1)))
    end
    return g
end

function boundary_message(ψIψ_bpc::BeliefPropagationCache, ψ::AbstractITensorNetwork, pe::PartitionEdge; rank::Int64 = 1,
    cache_update_kwargs = (; maxiter = 1))
    src_vertices, dst_vertices = vertices(ψIψ_bpc, src(pe)), vertices(ψIψ_bpc, dst(pe))
    src_col_vertices, dst_col_vertices =  unique(first.(src_vertices)), unique(first.(dst_vertices))

    red_src_col_vertices = filter(v -> !isempty(intersect(neighbors(ψ, v), dst_col_vertices)), src_col_vertices)
    internal_vertices = setdiff(src_col_vertices, red_src_col_vertices)
    g_r = merge_internal_vertices(underlying_graph(subgraph(ψ, src_col_vertices)), internal_vertices)
    pairs = [v => only(intersect(neighbors(ψ, v), dst_col_vertices)) for v in red_src_col_vertices]

    s = IndsNetwork(g_r)
    for p in pairs
        sind = only(linkinds(ψ, p))
        s[first(p)] = Index[sind, sind']
    end

    m = ITensorNetwork(v -> inds -> ITensor(1.0, inds), s; link_space = rank)
    m = rename_vertices(v -> (v, "message"), m)
    return m
end

function set_initial_messages(ψIψ_bpc::BeliefPropagationCache, ψ::AbstractITensorNetwork; rank::Int64 = 1)
    ms = messages(ψIψ_bpc)
    for pe in partitionedges(ψIψ_bpc)
        set!(ms, pe, boundary_message(ψIψ_bpc, ψ, pe; rank))
        set!(ms, reverse(pe), boundary_message(ψIψ_bpc, ψ, reverse(pe); rank))
    end
    return ψIψ_bpc
end

function initialize_cache(ψ::AbstractITensorNetwork; rank::Int64 = 1)
    Lx, Ly = maximum(vertices(ψ))
    ψIψ = QuadraticFormNetwork(ψ)
    vertex_groups = Lx >= Ly ? group(v -> first(first(v)), vertices(ψIψ)) : group(v -> last(first(v)), vertices(ψIψ))
    ψIψ_bpc = BeliefPropagationCache(ψIψ, vertex_groups)
    ψIψ_bpc = set_initial_messages(ψIψ_bpc, ψ; rank)

    return ψIψ_bpc
end

function get_column(ψIψ_bpc::BeliefPropagationCache, pv::PartitionVertex)
    verts = vertices(ψIψ_bpc, pv)
    return subgraph(tensornetwork(ψIψ_bpc), verts)
end

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

function merge_internal_vertices(g, internal_verts)
    g = copy(g)
    for v in internal_verts
      vns = neighbors(g, v)
      if !isempty(vns)
        g = merge_vertices(g, [first(vns), v])
      end
    end
    return g
end