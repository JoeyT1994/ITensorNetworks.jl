using NamedGraphs: NamedGraphs
using NamedGraphs.GraphsExtensions: add_edges
using ITensorNetworks: ITensorNetworks, BeliefPropagationCache, region_scalar
using ITensorNetworks.ITensorsExtensions: map_diag
using ITensorMPS: ITensorMPS, orthogonalize
using NamedGraphs.PartitionedGraphs:
  partitioned_graph, PartitionVertex, partitionvertex, partitioned_vertices, which_partition
using SplitApplyCombine: group
using ITensors: commoninds, random_itensor
using LinearAlgebra: pinv

struct BoundaryMPSCache{BPC,PG,PP} <: AbstractBeliefPropagationCache
  bp_cache::BPC
  partitionedplanargraph::PG
  partitionpair_partitionedges::PP
  maximum_virtual_dimension::Int64
end

bp_cache(bmpsc::BoundaryMPSCache) = bmpsc.bp_cache
partitionedplanargraph(bmpsc::BoundaryMPSCache) = bmpsc.partitionedplanargraph
ppg(bmpsc) = partitionedplanargraph(bmpsc)
maximum_virtual_dimension(bmpsc::BoundaryMPSCache) = bmpsc.maximum_virtual_dimension
partitionpair_partitionedges(bmpsc::BoundaryMPSCache) = bmpsc.partitionpair_partitionedges
planargraph(bmpsc::BoundaryMPSCache) = unpartitioned_graph(partitionedplanargraph(bmpsc))

function partitioned_tensornetwork(bmpsc::BoundaryMPSCache)
  return partitioned_tensornetwork(bp_cache(bmpsc))
end
messages(bmpsc::BoundaryMPSCache) = messages(bp_cache(bmpsc))

default_message_update_alg(bmpsc::BoundaryMPSCache) = "orthogonal"

function default_bp_maxiter(alg::Algorithm"orthogonal", bmpsc::BoundaryMPSCache)
  return default_bp_maxiter(partitioned_graph(ppg(bmpsc)))
end
default_bp_maxiter(alg::Algorithm"biorthogonal", bmpsc::BoundaryMPSCache) = 50
function default_edge_sequence(alg::Algorithm, bmpsc::BoundaryMPSCache)
  return pair.(default_edge_sequence(ppg(bmpsc)))
end
function default_message_update_kwargs(alg::Algorithm"orthogonal", bmpsc::BoundaryMPSCache)
  return (; niters=50, tolerance=1e-10)
end
function default_message_update_kwargs(
  alg::Algorithm"biorthogonal", bmpsc::BoundaryMPSCache
)
  return (; niters=3, tolerance=nothing)
end
default_boundarymps_message_rank(tn::AbstractITensorNetwork) = maxlinkdim(tn)^2
partitions(bmpsc::BoundaryMPSCache) = parent.(collect(partitionvertices(ppg(bmpsc))))
partitionpairs(bmpsc::BoundaryMPSCache) = pair.(partitionedges(ppg(bmpsc)))

function cache(
  alg::Algorithm"boundarymps",
  tn;
  bp_cache_construction_kwargs=default_cache_construction_kwargs(Algorithm("bp"), tn),
  kwargs...,
)
  return BoundaryMPSCache(
    BeliefPropagationCache(tn; bp_cache_construction_kwargs...); kwargs...
  )
end

function default_cache_construction_kwargs(alg::Algorithm"boundarymps", tn)
  return (;
    bp_cache_construction_kwargs=default_cache_construction_kwargs(Algorithm("bp"), tn)
  )
end

function default_cache_update_kwargs(alg::Algorithm"boundarymps")
  return (; alg="orthogonal", message_update_kwargs=(; niters=25, tolerance=1e-10))
end

function Base.copy(bmpsc::BoundaryMPSCache)
  return BoundaryMPSCache(
    copy(bp_cache(bmpsc)),
    copy(ppg(bmpsc)),
    copy(partitionpair_partitionedges(bmpsc)),
    maximum_virtual_dimension(bmpsc),
  )
end

function default_message(bmpsc::BoundaryMPSCache, pe::PartitionEdge; kwargs...)
  return default_message(bp_cache(bmpsc), pe::PartitionEdge; kwargs...)
end

function virtual_index_dimension(
  bmpsc::BoundaryMPSCache, pe1::PartitionEdge, pe2::PartitionEdge
)
  pes = planargraph_partitionpair_partitionedges(
    bmpsc, planargraph_partitionpair(bmpsc, pe1)
  )

  if findfirst(x -> x == pe1, pes) > findfirst(x -> x == pe2, pes)
    lower_pe, upper_pe = pe2, pe1
  else
    lower_pe, upper_pe = pe1, pe2
  end
  inds_above = reduce(vcat, linkinds.((bmpsc,), partitionedges_above(bmpsc, lower_pe)))
  inds_below = reduce(vcat, linkinds.((bmpsc,), partitionedges_below(bmpsc, upper_pe)))
  return minimum((
    prod(dim.(inds_above)), prod(dim.(inds_below)), maximum_virtual_dimension(bmpsc)
  ))
end

function planargraph_vertices(bmpsc::BoundaryMPSCache, partition)
  return vertices(ppg(bmpsc), PartitionVertex(partition))
end
function planargraph_partition(bmpsc::BoundaryMPSCache, vertex)
  return parent(partitionvertex(ppg(bmpsc), vertex))
end
function planargraph_partitions(bmpsc::BoundaryMPSCache, verts)
  return parent.(partitionvertices(ppg(bmpsc), verts))
end
function planargraph_partitionpair(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  return pair(partitionedge(ppg(bmpsc), parent(pe)))
end
function planargraph_partitionpair_partitionedges(
  bmpsc::BoundaryMPSCache, partition_pair::Pair
)
  return partitionpair_partitionedges(bmpsc)[partition_pair]
end

function BoundaryMPSCache(
  bpc::BeliefPropagationCache;
  grouping_function::Function=v -> first(v),
  group_sorting_function::Function=v -> last(v),
  message_rank::Int64=default_boundarymps_message_rank(tensornetwork(bpc)),
)
  bpc = insert_pseudo_planar_edges(bpc; grouping_function)
  planar_graph = partitioned_graph(bpc)
  vertex_groups = group(grouping_function, collect(vertices(planar_graph)))
  vertex_groups = map(x -> sort(x; by=group_sorting_function), vertex_groups)
  ppg = PartitionedGraph(planar_graph, vertex_groups)
  partitionpairs = vcat(pair.(partitionedges(ppg)), reverse.(pair.(partitionedges(ppg))))
  pp_pe = Dictionary(partitionpairs, sorted_partitionedges.((ppg,), partitionpairs))
  bmpsc = BoundaryMPSCache(bpc, ppg, pp_pe, message_rank)
  return set_interpartition_messages(bmpsc)
end

function BoundaryMPSCache(tn, args...; kwargs...)
  return BoundaryMPSCache(BeliefPropagationCache(tn, args...); kwargs...)
end

#Get all partitionedges between the pair of neighboring partitions, sorted 
#by the position of the source in the pg
function sorted_partitionedges(pg::PartitionedGraph, partitionpair::Pair)
  src_vs, dst_vs = vertices(pg, PartitionVertex(first(partitionpair))),
  vertices(pg, PartitionVertex(last(partitionpair)))
  es = reduce(
    vcat,
    [
      [src_v => dst_v for dst_v in intersect(neighbors(pg, src_v), dst_vs)] for
      src_v in src_vs
    ],
  )
  es = sort(NamedEdge.(es); by=x -> findfirst(isequal(src(x)), src_vs))
  return PartitionEdge.(es)
end

#Functions to get the parellel partitionedges sitting above and below a partitionedge
function partitionedges_above(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  pes = planargraph_partitionpair_partitionedges(
    bmpsc, planargraph_partitionpair(bmpsc, pe)
  )
  pe_pos = only(findall(x -> x == pe, pes))
  return PartitionEdge[pes[i] for i in (pe_pos + 1):length(pes)]
end

function partitionedges_below(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  pes = planargraph_partitionpair_partitionedges(
    bmpsc, planargraph_partitionpair(bmpsc, pe)
  )
  pe_pos = only(findall(x -> x == pe, pes))
  return PartitionEdge[pes[i] for i in 1:(pe_pos - 1)]
end

function partitionedge_above(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  pes_above = partitionedges_above(bmpsc, pe)
  isempty(pes_above) && return nothing
  return first(pes_above)
end

function partitionedge_below(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  pes_below = partitionedges_below(bmpsc, pe)
  isempty(pes_below) && return nothing
  return last(pes_below)
end

#Get the sequence of pairs partitionedges that need to be updated to move the MPS gauge from pe1 to pe2
function mps_gauge_update_sequence(
  bmpsc::BoundaryMPSCache, pe1::Union{Nothing,PartitionEdge}, pe2::PartitionEdge
)
  isnothing(pe1) && return mps_gauge_update_sequence(bmpsc, pe2)
  ppgpe1, ppgpe2 = planargraph_partitionpair(bmpsc, pe1),
  planargraph_partitionpair(bmpsc, pe2)
  @assert ppgpe1 == ppgpe2
  pes = planargraph_partitionpair_partitionedges(bmpsc, ppgpe1)
  return pair_sequence(pes, pe1, pe2)
end

#Get the sequence of pairs partitionedges that need to be updated to move the MPS gauge onto pe
function mps_gauge_update_sequence(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  ppgpe = planargraph_partitionpair(bmpsc, pe)
  pes = planargraph_partitionpair_partitionedges(bmpsc, ppgpe)
  return vcat(
    mps_gauge_update_sequence(bmpsc, last(pes), pe),
    mps_gauge_update_sequence(bmpsc, first(pes), pe),
  )
end

#Initialise all the message tensors for the pairs of neighboring partitions, with virtual rank given by message rank
function set_interpartition_messages(
  bmpsc::BoundaryMPSCache, partitionpairs::Vector{<:Pair}
)
  bmpsc = copy(bmpsc)
  ms = messages(bmpsc)
  for partitionpair in partitionpairs
    pes = planargraph_partitionpair_partitionedges(bmpsc, partitionpair)
    for pe in pes
      set!(ms, pe, ITensor[dense(delta(linkinds(bmpsc, pe)))])
    end
    for i in 1:(length(pes) - 1)
      virt_dim = virtual_index_dimension(bmpsc, pes[i], pes[i + 1])
      ind = Index(virt_dim, "m$(i)$(i+1)")
      m1, m2 = only(ms[pes[i]]), only(ms[pes[i + 1]])
      set!(ms, pes[i], ITensor[m1 * delta(ind)])
      set!(ms, pes[i + 1], ITensor[m2 * delta(ind)])
    end
  end
  return bmpsc
end

#Initialise all the interpartition message tensors with virtual rank given by message rank
function set_interpartition_messages(bmpsc::BoundaryMPSCache)
  partitionpairs = pair.(partitionedges(ppg(bmpsc)))
  return set_interpartition_messages(bmpsc, vcat(partitionpairs, reverse.(partitionpairs)))
end

#Switch the message on partition edge pe with its reverse (and dagger them)
function switch_message(bmpsc::BoundaryMPSCache, pe::PartitionEdge)
  bmpsc = copy(bmpsc)
  ms = messages(bmpsc)
  me, mer = message(bmpsc, pe), message(bmpsc, reverse(pe))
  set!(ms, pe, dag.(mer))
  set!(ms, reverse(pe), dag.(me))
  return bmpsc
end

#Switch the message tensors from partitionpair  i -> i + 1 with those from i + 1 -> i
function switch_messages(bmpsc::BoundaryMPSCache, partitionpair::Pair)
  for pe in planargraph_partitionpair_partitionedges(bmpsc, partitionpair)
    bmpsc = switch_message(bmpsc, pe)
  end
  return bmpsc
end

#Update all messages tensors within a partition
function partition_update(bmpsc::BoundaryMPSCache, partition)
  vs = planargraph_vertices(bmpsc, partition)
  bmpsc = partition_update(bmpsc, first(vs), last(vs))
  bmpsc = partition_update(bmpsc, last(vs), first(vs))
  return bmpsc
end

function partition_update_sequence(bmpsc::BoundaryMPSCache, v1, v2)
  isnothing(v1) && return partition_update_sequence(bmpsc, v2)
  pv = planargraph_partition(bmpsc, v1)
  g = subgraph(unpartitioned_graph(ppg(bmpsc)), planargraph_vertices(bmpsc, pv))
  return PartitionEdge.(a_star(g, v1, v2))
end
function partition_update_sequence(bmpsc::BoundaryMPSCache, v)
  pv = planargraph_partition(bmpsc, v)
  g = subgraph(unpartitioned_graph(ppg(bmpsc)), planargraph_vertices(bmpsc, pv))
  return PartitionEdge.(post_order_dfs_edges(g, v))
end

#Update all messages within a partition along the path from from v1 to v2
function partition_update(bmpsc::BoundaryMPSCache, args...)
  return update(
    Algorithm("simplebp"),
    bmpsc,
    partition_update_sequence(bmpsc, args...);
    message_update_function_kwargs=(; normalize=false),
  )
end

#Move the orthogonality centre one step on an interpartition from the message tensor on pe1 to that on pe2 
function gauge_step(
  alg::Algorithm"orthogonal",
  bmpsc::BoundaryMPSCache,
  pe1::PartitionEdge,
  pe2::PartitionEdge;
  kwargs...,
)
  bmpsc = copy(bmpsc)
  ms = messages(bmpsc)
  m1, m2 = only(message(bmpsc, pe1)), only(message(bmpsc, pe2))
  @assert !isempty(commoninds(m1, m2))
  left_inds = uniqueinds(m1, m2)
  m1, Y = factorize(m1, left_inds; ortho="left", kwargs...)
  m2 = m2 * Y
  set!(ms, pe1, ITensor[m1])
  set!(ms, pe2, ITensor[m2])
  return bmpsc
end

#Move the biorthogonality centre one step on an interpartition from the partition edge pe1 (and its reverse) to that on pe2 
function gauge_step(
  alg::Algorithm"biorthogonal",
  bmpsc::BoundaryMPSCache,
  pe1::PartitionEdge,
  pe2::PartitionEdge,
)
  bmpsc = copy(bmpsc)
  ms = messages(bmpsc)

  m1, m1r = only(message(bmpsc, pe1)), only(message(bmpsc, reverse(pe1)))
  m2, m2r = only(message(bmpsc, pe2)), only(message(bmpsc, reverse(pe2)))
  top_cind, bottom_cind = commonind(m1, m2), commonind(m1r, m2r)
  m1_siteinds, m2_siteinds = commoninds(m1, m1r), commoninds(m2, m2r)
  top_ncind = setdiff(inds(m1), [m1_siteinds; top_cind])
  bottom_ncind = setdiff(inds(m1r), [m1_siteinds; bottom_cind])

  E = if isempty(top_ncind)
    m1 * m1r
  else
    m1 * replaceind(m1r, only(bottom_ncind), only(top_ncind))
  end
  U, S, V = svd(E, bottom_cind; alg="recursive")

  S_sqrtinv = map_diag(x -> pinv(sqrt(x)), S)
  S_sqrt = map_diag(x -> sqrt(x), S)

  m1 = replaceind((m1 * dag(V)) * S_sqrtinv, commonind(U, S), top_cind)
  m1r = replaceind((m1r * dag(U)) * S_sqrtinv, commonind(V, S), bottom_cind)
  m2 = replaceind((m2 * V) * S_sqrt, commonind(U, S), top_cind)
  m2r = replaceind((m2r * U) * S_sqrt, commonind(V, S), bottom_cind)
  set!(ms, pe1, ITensor[m1])
  set!(ms, reverse(pe1), ITensor[m1r])
  set!(ms, pe2, ITensor[m2])
  set!(ms, reverse(pe2), ITensor[m2r])

  return bmpsc
end

#Move the orthogonality / biorthogonality centre on an interpartition via a sequence of steps between message tensors
function gauge_walk(alg::Algorithm, bmpsc::BoundaryMPSCache, seq::Vector; kwargs...)
  for (pe1, pe2) in seq
    bmpsc = gauge_step(alg::Algorithm, bmpsc, pe1, pe2; kwargs...)
  end
  return bmpsc
end

function gauge(alg::Algorithm, bmpsc::BoundaryMPSCache, args...; kwargs...)
  return gauge_walk(alg, bmpsc, mps_gauge_update_sequence(bmpsc, args...); kwargs...)
end

#Move the orthogonality centre on an interpartition to the message tensor on pe or between two pes
function ITensorMPS.orthogonalize(bmpsc::BoundaryMPSCache, args...; kwargs...)
  return gauge(Algorithm("orthogonal"), bmpsc, args...; kwargs...)
end

#Move the biorthogonality centre on an interpartition to the message tensor or between two pes
function biorthogonalize(bmpsc::BoundaryMPSCache, args...; kwargs...)
  return gauge(Algorithm("biorthogonal"), bmpsc, args...; kwargs...)
end

function default_inserter(
  alg::Algorithm"orthogonal",
  bmpsc::BoundaryMPSCache,
  pe::PartitionEdge,
  me::Vector{ITensor},
)
  return set_message(bmpsc, reverse(pe), dag.(me))
end

function default_inserter(
  alg::Algorithm"biorthogonal",
  bmpsc::BoundaryMPSCache,
  pe::PartitionEdge,
  me::Vector{ITensor},
)
  p_above, p_below = partitionedge_above(bmpsc, pe), partitionedge_below(bmpsc, pe)
  me = only(me)
  me_prev = only(message(bmpsc, pe))
  for pe in filter(x -> !isnothing(x), [p_above, p_below])
    ind1 = commonind(me, only(message(bmpsc, reverse(pe))))
    ind2 = commonind(me_prev, only(message(bmpsc, pe)))
    me *= delta(ind1, ind2)
  end
  return set_message(bmpsc, pe, ITensor[me])
end

function default_updater(
  alg::Algorithm"orthogonal", bmpsc::BoundaryMPSCache, prev_pe, update_pe, prev_v, cur_v
)
  rev_prev_pe = isnothing(prev_pe) ? nothing : reverse(prev_pe)
  bmpsc = gauge(alg, bmpsc, rev_prev_pe, reverse(update_pe))
  bmpsc = partition_update(bmpsc, prev_v, cur_v)
  return bmpsc
end

function default_updater(
  alg::Algorithm"biorthogonal", bmpsc::BoundaryMPSCache, prev_pe, update_pe, prev_v, cur_v
)
  bmpsc = gauge(alg, bmpsc, prev_pe, update_pe)
  bmpsc = partition_update(bmpsc, prev_v, cur_v)
  return bmpsc
end

function default_cache_prep_function(
  alg::Algorithm"biorthogonal", bmpsc::BoundaryMPSCache, partitionpair
)
  return bmpsc
end
function default_cache_prep_function(
  alg::Algorithm"orthogonal", bmpsc::BoundaryMPSCache, partitionpair
)
  return switch_messages(bmpsc, partitionpair)
end

default_niters(alg::Algorithm"orthogonal") = 25
default_niters(alg::Algorithm"biorthogonal") = 3
default_tolerance(alg::Algorithm"orthogonal") = 1e-10
default_tolerance(alg::Algorithm"biorthogonal") = nothing

function default_costfunction(
  alg::Algorithm"orthogonal",
  bmpsc::BoundaryMPSCache,
  pe::PartitionEdge,
  me::Vector{ITensor},
)
  return region_scalar(bp_cache(bmpsc), src(pe)) / norm(only(me))
end

function default_costfunction(
  alg::Algorithm"biorthogonal",
  bmpsc::BoundaryMPSCache,
  pe::PartitionEdge,
  me::Vector{ITensor},
)
  return region_scalar(bp_cache(bmpsc), src(pe)) /
         dot(only(me), only(message(bmpsc, reverse(pe))))
end

#Update all the message tensors on an interpartition via a specified fitting procedure 
#TODO: Make two-site possible
function update(
  alg::Algorithm,
  bmpsc::BoundaryMPSCache,
  partitionpair::Pair;
  inserter=default_inserter,
  costfunction=default_costfunction,
  updater=default_updater,
  cache_prep_function=default_cache_prep_function,
  niters::Int64=default_niters(alg),
  tolerance=default_tolerance(alg),
  normalize=true,
)
  bmpsc = cache_prep_function(alg, bmpsc, partitionpair)
  pes = planargraph_partitionpair_partitionedges(bmpsc, partitionpair)
  update_seq = vcat(pes, reverse(pes)[2:length(pes)])
  prev_v, prev_pe = nothing, nothing
  prev_cf = 0
  for i in 1:niters
    cf = 0
    for update_pe in update_seq
      cur_v = parent(src(update_pe))
      bmpsc = updater(alg, bmpsc, prev_pe, update_pe, prev_v, cur_v)
      me = updated_message(bmpsc, update_pe; message_update_function_kwargs=(; normalize))
      cf += costfunction(alg, bmpsc, update_pe, me)
      bmpsc = inserter(alg, bmpsc, update_pe, me)
      prev_v, prev_pe = cur_v, update_pe
    end
    epsilon = abs(cf - prev_cf) / length(update_seq)
    if !isnothing(tolerance) && epsilon < tolerance
      return cache_prep_function(alg, bmpsc, partitionpair)
    else
      prev_cf = cf
    end
  end
  return cache_prep_function(alg, bmpsc, partitionpair)
end

#Assume all vertices live in the same partition for now
function ITensorNetworks.environment(bmpsc::BoundaryMPSCache, verts::Vector; kwargs...)
  vs = parent.((partitionvertices(bp_cache(bmpsc), verts)))
  pv = only(planargraph_partitions(bmpsc, vs))
  bmpsc = partition_update(bmpsc, pv)
  return environment(bp_cache(bmpsc), verts; kwargs...)
end

function region_scalar(bmpsc::BoundaryMPSCache, partition)
  partition_vs = planargraph_vertices(bmpsc, partition)
  bmpsc = partition_update(bmpsc, first(partition_vs), last(partition_vs))
  return region_scalar(bp_cache(bmpsc), PartitionVertex(last(partition_vs)))
end

function region_scalar(bmpsc::BoundaryMPSCache, partitionpair::Pair)
  pes = planargraph_partitionpair_partitionedges(bmpsc, partitionpair)
  out = ITensor(1.0)
  for pe in pes
    out = (out * (only(message(bmpsc, pe)))) * only(message(bmpsc, reverse(pe)))
  end
  return out[]
end
