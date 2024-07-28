using ITensorNetworks: BeliefPropagationCache, messages, partitioned_tensornetwork
using NamedGraphs.PartitionedGraphs: PartitionEdge, partitionedges, partitionvertices
using NamedGraphs.NamedGraphGenerators: named_hexagonal_lattice_graph

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