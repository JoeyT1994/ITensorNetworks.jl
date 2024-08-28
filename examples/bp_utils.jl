function ITensorMPS.MPS(ψ::ITensorNetwork)
    return MPS([ψ[v] for v in vertices(ψ)])
end

function update_message_planar(ψIψ_bpc::BeliefPropagationCache, pe::PartitionEdge;
    kwargs...)
    pe_in = setdiff(boundary_partitionedges(ψIψ_bpc, src(pe); dir=:in), [reverse(pe)])
    if !isempty(pe_in)
        Φ, A, Ψ = message(ψIψ_bpc, only(pe_in)), get_column(ψIψ_bpc, src(pe)), message(ψIψ_bpc, pe)
    else
        Φ, A, Ψ = nothing, get_column(ψIψ_bpc, src(pe)), message(ψIψ_bpc, pe)
    end
    Ψnew = optimise(Φ, A, Ψ; kwargs...)

    return Ψnew
end

"""
Do a sequential update of the message tensors on `edges`
"""
function update_planar(
  bp_cache::BeliefPropagationCache,
  edges::Vector{<:PartitionEdge};
  (update_diff!)=nothing,
  kwargs...,
)
  bp_cache_updated = copy(bp_cache)
  mts = messages(bp_cache_updated)
  for e in edges
    set!(mts, e, update_message_planar(bp_cache_updated, e; kwargs...))
    if !isnothing(update_diff!)
        f = inner(MPS(message(bp_cache, e)), MPS(mts[e]))
        update_diff![] += 1.0  - f*conj(f)
    end
  end
  return bp_cache_updated
end

"""
More generic interface for update, with default params
"""
function update_planar(
  bp_cache::BeliefPropagationCache;
  edges=default_edge_sequence(bp_cache),
  maxiter=default_bp_maxiter(bp_cache),
  tol=nothing,
  verbose=false,
  kwargs...,
)
  compute_error = !isnothing(tol)
  if isnothing(maxiter)
    error("You need to specify a number of iterations for BP!")
  end
  for i in 1:maxiter
    diff = compute_error ? Ref(0.0) : nothing
    bp_cache = update_planar(bp_cache, edges; (update_diff!)=diff, kwargs...)
    @show (diff.x / length(edges))
    if compute_error && (diff.x / length(edges)) <= tol
      if verbose
        println("BP converged to desired precision after $i iterations.")
      end
      break
    end
  end
  return bp_cache
end

function boundary_message(ψIψ_bpc::BeliefPropagationCache, ψ::AbstractITensorNetwork, pe::PartitionEdge; rank::Int64 = 1,
    cache_update_kwargs = (; maxiter = 1))
    src_vertices, dst_vertices = vertices(ψIψ_bpc, src(pe)), vertices(ψIψ_bpc, dst(pe))
    src_col_vertices, dst_col_vertices =  unique(first.(src_vertices)), unique(first.(dst_vertices))

    vs = filter(v -> !isempty(intersect(neighbors(ψ, v), dst_col_vertices)), src_col_vertices)
    #Base this on whether Lx >= Ly
    vs = sort(vs; by = v -> last(v))
    g_r = NamedGraph(vs)
    g_r = add_edges(g_r, [NamedEdge(vs[i] =>vs[i+1]) for i in 1:(length(vs)-1)])
    if has_edge(ψ, NamedEdge(first(vs) => last(vs)))
        g_r = add_edge(g_r, NamedEdge(first(vs) => last(vs)))
    end
    pairs = [v => only(intersect(neighbors(ψ, v), dst_col_vertices)) for v in vs]

    s = IndsNetwork(g_r)
    for p in pairs
        sind = only(linkinds(ψ, p))
        s[first(p)] = Index[sind, sind']
    end

    m = ITensorNetwork(v -> inds -> ITensor(1.0, inds), s; link_space = rank)
    m = rename_vertices(v -> (v, "message"), m)
    m = orthogonalize(m, first(vertices(m)))
    m[(first(vertices(m)))] = normalize(m[(first(vertices(m)))])
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

function build_sandwich(ψIψ_bpc::BeliefPropagationCache, pv)
    pes = setdiff(boundary_partitionedges(ψIψ_bpc, pv; dir=:in))
    if length(pes) == 1
        Φ, A, ψ = message(ψIψ_bpc, only(pes)), get_column(ψIψ_bpc, pv), nothing
    else
        Φ, A, ψ = message(ψIψ_bpc, first(pes)), get_column(ψIψ_bpc, pv), message(ψIψ_bpc, last(pes))
    end
    return build_sandwich(Φ, A, ψ)
end

function build_sandwich(ψ::ITensorNetwork, A::ITensorNetwork, ϕ)
    if ϕ != nothing
        ϕAψ = disjoint_union("operator" => A,"bra" => ϕ,"ket" => ψ)
    else
        ϕAψ = disjoint_union("operator" => A,"ket" => ψ)
    end
    ptn = PartitionedGraph(ϕAψ, group(v -> last(first(first(v))), vertices(ϕAψ)))
    g, pg, pvs, wp = unpartitioned_graph(ptn), partitioned_graph(ptn), partitioned_vertices(ptn), which_partition(ptn)
    vs = sort(collect(vertices(pg)))
    pg = rem_edges(pg, edges(pg))
    pg = add_edges(pg, [NamedEdge(vs[i] => vs[i+1]) for i in 1:(length(vs)-1)])
    ptn = PartitionedGraph(g, pg, pvs, wp)
    return BeliefPropagationCache(ptn)
end
    

function expect_planar(ψIψ_bpc::BeliefPropagationCache, s::IndsNetwork, op::String, v)
    pv = only(partitionvertices(ψIψ_bpc, [(v, "operator")]))
    ϕAψ_bpc = build_sandwich(ψIψ_bpc, pv)
    seq = PartitionEdge.(post_order_dfs_edges(partitioned_graph(ϕAψ_bpc), parent(pv)))
    ϕAψ_bpc = update(ϕAψ_bpc, seq; message_update = ms -> default_message_update(ms; normalize = false))
    denom = region_scalar(ϕAψ_bpc, pv)

    op = ITensors.op(op, s[v])
    ϕOψ_bpc = update_factor(ψIψ_bpc, (v, "operator"), op)
    ϕOψ_bpc = build_sandwich(ϕOψ_bpc, pv)
    seq = PartitionEdge.(post_order_dfs_edges(partitioned_graph(ϕOψ_bpc), parent(pv)))
    ϕOψ_bpc = update(ϕOψ_bpc, seq; message_update = ms -> default_message_update(ms; normalize = false))
    numer = region_scalar(ϕOψ_bpc, pv)
    return numer / denom
end

function expect_planar_exact(ψIψ_bpc::BeliefPropagationCache, s::IndsNetwork, op::String, v)
    pv = only(partitionvertices(ψIψ_bpc, [(v, "operator")]))
    tn = tensornetwork(ψIψ_bpc)
    seq = contraction_sequence(tn; alg = "sa_bipartite")
    denom = contract(tn; sequence = seq)[]

    op = ITensors.op(op, s[v])
    ϕOψ_bpc = update_factor(ψIψ_bpc, (v, "operator"), op)
    tn = tensornetwork(ϕOψ_bpc)
    seq = contraction_sequence(tn; alg = "sa_bipartite")
    numer = contract(tn; sequence = seq)[]
    
    return numer / denom
end