#using FillArrays: OneElement
#using DataGraphs: DataGraph
using Graphs: degree, is_tree
using ITensors: flux, has_fermion_string, itensor, removeqns, space
using ITensors.LazyApply: Prod, Sum, coefficient
using ITensors.NDTensors: Block, blockdim, maxdim, nblocks, nnzblocks, truncate!
using ITensors.Ops: argument, coefficient, Op, OpSum, name, params, site, terms, which_op
using NamedGraphs.GraphsExtensions:
  GraphsExtensions, boundary_edges, degrees, is_leaf_vertex, vertex_path
using StaticArrays: MVector

# 
# Utility methods
# 

function align_edges(edges, reference_edges)
  return intersect(Iterators.flatten(zip(edges, reverse.(edges))), reference_edges)
end

function align_and_reorder_edges(edges, reference_edges)
  return intersect(reference_edges, align_edges(edges, reference_edges))
end

function split_at_vertex(g::AbstractGraph, v)
  g = copy(g)
  rem_vertex!(g, v)
  return Set.(connected_components(g))
end

# 
# Tree adaptations of functionalities in ITensors.jl/src/physics/autompo/opsum_to_mpo.jl
# 

function determine_coefficient_type(terms)
  isempty(terms) && return Float64
  if all(t -> isreal(coefficient(t)), terms)
    return real(typeof(coefficient(first(terms))))
  end
  return typeof(coefficient(first(terms)))
end

"""
    ttn_svd(os::OpSum, sites::IndsNetwork, root_vertex, kwargs...)

Construct a TreeTensorNetwork from a symbolic OpSum representation of a
Hamiltonian, compressing shared interaction channels.
"""
function ttn_svd(os::OpSum, sites::IndsNetwork, root_vertex; kwargs...)
  # Function barrier to improve type stability
  coefficient_type = determine_coefficient_type(terms(os))
  return ttn_svd(coefficient_type, os, sites, root_vertex; kwargs...)
end

# TODO: should be equivalent to `sort!(v); unique!(v)` however,
# Base.unique! is giving incorrect results for data involving the 
# Prod type from LazyApply. This could be a combination of Base.unique!
# not strictly relying on isequal combined with an incorrect
# implementation of isequal/isless for Prod.
function _sort_unique!(v::Vector)
  N = length(v)
  (N == 0) && return nothing
  sort!(v)
  n = 1
  u = 2
  while u <= N
    while u < N && v[u] == v[n]
      u += 1
    end
    if v[u] != v[n]
      v[n + 1] = v[u]
      n += 1
    end
    u += 1
  end
  resize!(v, n)
  return nothing
end

function pos_in_link!(linkmap::Dict, k)
  isempty(k) && return -1
  pos = get(linkmap, k, -1)
  if pos == -1
    pos = length(linkmap) + 1
    linkmap[k] = pos
  end
  return pos
end

function make_symbolic_ttn(
  coefficient_type,
  opsum::OpSum,
  sites::IndsNetwork;
  ordered_verts,
  ordered_edges,
  root_vertex,
  term_qn_map,
)
  inmaps = Dict{Pair{edgetype(sites),QN},Dict{Vector{Op},Int}}()
  outmaps = Dict{Pair{edgetype(sites),QN},Dict{Vector{Op},Int}}()

  #g = underlying_graph(sites)

  # Bond coefficients for incoming edge channels.
  # These become the "M" coefficient matrices that get SVD'd.
  inbond_coefs = Dict(
    e => Dict{QN,Vector{MatElem{coefficient_type}}}() for e in ordered_edges
  )

  # List of terms for which the coefficient has been added to a site factor
  site_coef_done = Prod{Op}[]

  # Temporary symbolic representation of TTN Hamiltonian
  symbolic_ttn = Dict(
    v => QNArrElem{Scaled{coefficient_type,Prod{Op}},degree(sites, v)}[] for
    v in ordered_verts
  )

  # Build compressed finite state machine representation (symbolic_ttn)
  for v in ordered_verts
    v_degree = degree(sites, v)
    # For every vertex, find all edges that contain this vertex
    # (align_and_reorder_edges makes the output of indicident edges match the
    #  direction and ordering match that of ordered_edges)
    edges = align_and_reorder_edges(incident_edges(sites, v), ordered_edges)

    # Use the corresponding ordering as index order for tensor elements at this site
    dim_in = findfirst(e -> dst(e) == v, edges)
    edge_in = (isnothing(dim_in) ? nothing : edges[dim_in])
    dims_out = findall(e -> src(e) == v, edges)
    edges_out = edges[dims_out]

    # For every site v' except v, determine the incident edge to v
    # that lies in the edge_path(v',v)
    # TODO: better way to make which_incident_edge below?
    subgraphs = split_at_vertex(sites, v)
    _boundary_edges = [
      only(boundary_edges(underlying_graph(sites), subgraph)) for subgraph in subgraphs
    ]
    _boundary_edges = align_edges(_boundary_edges, edges)
    which_incident_edge = Dict(
      Iterators.flatten([
        subgraphs[i] .=> ((_boundary_edges[i]),) for i in eachindex(subgraphs)
      ]),
    )

    # Sanity check, leaves only have single incoming or outgoing edge
    @assert !isempty(dims_out) || !isnothing(dim_in)
    (isempty(dims_out) || isnothing(dim_in)) && @assert is_leaf_vertex(sites, v)

    for term in opsum
      # Loop over OpSum and pick out terms that act on current vertex
      ops = ITensors.terms(term)
      if v in ITensors.site.(ops)
        crosses_vertex = true
      else
        crosses_vertex =
          !isone(length(Set([which_incident_edge[site] for site in site.(ops)])))
      end
      # If term doesn't cross vertex, skip it
      crosses_vertex || continue

      # Filter out ops that acts on current vertex
      onsite_ops = filter(t -> (site(t) == v), ops)
      non_onsite_ops = setdiff(ops, onsite_ops)

      # Filter out ops that come in from the direction of the incoming edge
      incoming_ops = filter(t -> which_incident_edge[site(t)] == edge_in, non_onsite_ops)

      # Also store all non-incoming ops in standard order, used for channel merging
      non_incoming_ops = filter(
        t -> (site(t) == v) || which_incident_edge[site(t)] != edge_in, ops
      )

      # For every outgoing edge, filter out ops that go out along that edge
      outgoing_ops = Dict(
        e => filter(t -> which_incident_edge[site(t)] == e, non_onsite_ops) for
        e in edges_out
      )

      # Compute QNs
      incoming_qn = term_qn_map(incoming_ops)
      non_incoming_qn = term_qn_map(non_incoming_ops)
      site_qn = term_qn_map(onsite_ops)

      # Initialize QNArrayElement indices and quantum numbers 
      T_inds = MVector{v_degree}(fill(-1, v_degree))
      T_qns = MVector{v_degree}(fill(QN(), v_degree))
      # initialize ArrayElement indices for inbond_coefs
      bond_row = -1
      bond_col = -1
      if !isempty(incoming_ops)
        # Get the correct map from edge=>QN to term and channel.
        # This checks if term exists on edge=>QN (otherwise insert it) and returns its index.
        coutmap = get!(outmaps, edge_in => non_incoming_qn, Dict{Vector{Op},Int}())
        cinmap = get!(inmaps, edge_in => -incoming_qn, Dict{Vector{Op},Int}())

        bond_row = pos_in_link!(cinmap, incoming_ops)
        bond_col = pos_in_link!(coutmap, non_incoming_ops) # get incoming channel
        bond_coef = convert(coefficient_type, coefficient(term))
        q_inbond_coefs = get!(
          inbond_coefs[edge_in], incoming_qn, MatElem{coefficient_type}[]
        )
        push!(q_inbond_coefs, MatElem(bond_row, bond_col, bond_coef))
        T_inds[dim_in] = bond_col
        T_qns[dim_in] = -incoming_qn
      end
      for dout in dims_out
        out_edge = edges[dout]
        out_op = outgoing_ops[out_edge]
        coutmap = get!(outmaps, out_edge => term_qn_map(out_op), Dict{Vector{Op},Int}())
        # Add outgoing channel
        T_inds[dout] = pos_in_link!(coutmap, out_op)
        T_qns[dout] = term_qn_map(out_op)
      end
      # If term starts at this site, add its coefficient as a site factor
      site_coef = one(coefficient_type)
      if (isnothing(dim_in) || T_inds[dim_in] == -1) && argument(term) ∉ site_coef_done
        site_coef = convert(coefficient_type, coefficient(term))
        push!(site_coef_done, argument(term))
      end
      # Add onsite identity for interactions passing through vertex
      if isempty(onsite_ops)
        if !ITensors.using_auto_fermion() && isfermionic(incoming_ops, sites)
          push!(onsite_ops, Op("F", v))
        else
          push!(onsite_ops, Op("Id", v))
        end
      end
      # Save indices and value of symbolic tensor entry
      el = QNArrElem(T_qns, T_inds, site_coef * Prod(onsite_ops))
      push!(symbolic_ttn[v], el)
    end
    _sort_unique!(symbolic_ttn[v])
  end

  return symbolic_ttn, inbond_coefs
end

function svd_bond_coefs(
  coefficient_type, sites, inbond_coefs; ordered_verts, ordered_edges, kws...
)
  Vs = Dict(e => Dict{QN,Matrix{coefficient_type}}() for e in ordered_edges)
  for v in ordered_verts
    edges = align_and_reorder_edges(incident_edges(sites, v), ordered_edges)
    dim_in = findfirst(e -> dst(e) == v, edges)
    if !isnothing(dim_in) && !isempty(inbond_coefs[edges[dim_in]])
      for (q, mat) in inbond_coefs[edges[dim_in]]
        M = toMatrix(mat)
        U, S, V = svd(M)
        P = S .^ 2
        truncate!(P; kws...)
        tdim = length(P)
        nc = size(M, 2)
        Vs[edges[dim_in]][q] = Matrix{coefficient_type}(V[1:nc, 1:tdim])
      end
    end
  end
  return Vs
end

function compress_ttn(
  coefficient_type, sites0, Hflux, symbolic_ttn, Vs; ordered_verts, ordered_edges
)
  # Insert dummy indices on internal vertices, these will not show up in the final tensor
  # TODO: come up with a better solution for this
  sites = copy(sites0)
  is_internal = Dict{vertextype(sites),Bool}()
  for v in ordered_verts
    is_internal[v] = isempty(sites[v])
    if isempty(sites[v])
      # FIXME: This logic only works for trivial flux, breaks for nonzero flux
      # TODO: add assert or fix and add test!
      sites[v] = [Index(Hflux => 1)]
    end
  end

  linkdir_ref = ITensors.In  # safe to always use autofermion default here
  # Compress this symbolic_ttn representation into dense form
  thishasqns = any(v -> hasqns(sites[v]), vertices(sites))

  link_space = Dict{edgetype(sites),Index}()
  for e in ordered_edges
    operator_blocks = [q => size(Vq, 2) for (q, Vq) in Vs[e]]
    link_space[e] = Index(
      QN() => 1, operator_blocks..., Hflux => 1; tags=edge_tag(e), dir=linkdir_ref
    )
  end

  H = ttn(sites0)   # initialize TTN without the dummy indices added
  function qnblock(i::Index, q::QN)
    for b in 2:(nblocks(i) - 1)
      flux(i, Block(b)) == q && return b
    end
    return error("Could not find block of QNIndex with matching QN")
  end
  qnblockdim(i::Index, q::QN) = blockdim(i, qnblock(i, q))

  for v in ordered_verts
    v_degree = degree(sites, v)
    # Redo the whole thing like before
    # TODO: use neighborhood instead of going through all edges, see above
    edges = align_and_reorder_edges(incident_edges(sites, v), ordered_edges)
    dim_in = findfirst(e -> dst(e) == v, edges)
    dims_out = findall(e -> src(e) == v, edges)
    # slice isometries at this vertex
    Vv = [Vs[e] for e in edges]
    linkinds = [link_space[e] for e in edges]

    # construct blocks
    blocks = Dict{Tuple{Block{v_degree},Vector{Op}},Array{coefficient_type,v_degree}}()
    for el in symbolic_ttn[v]
      t = el.val
      (abs(coefficient(t)) > eps(real(coefficient_type))) || continue
      block_helper_inds = fill(-1, v_degree) # we manipulate T_inds later, and loose track of ending/starting information, so keep track of it here
      T_inds = el.idxs
      T_qns = el.qn_idxs
      ct = convert(coefficient_type, coefficient(t))
      sublinkdims = [
        (T_inds[i] == -1 ? 1 : qnblockdim(linkinds[i], T_qns[i])) for i in 1:v_degree
      ]
      zero_arr() = zeros(coefficient_type, sublinkdims...)
      terminal_dims = findall(d -> T_inds[d] == -1, 1:v_degree)   # directions in which term starts or ends
      normal_dims = findall(d -> T_inds[d] ≠ -1, 1:v_degree)      # normal dimensions, do truncation thingies
      T_inds[terminal_dims] .= 1                                  # start in channel 1  ###??
      block_helper_inds[terminal_dims] .= 1
      for dout in filter(d -> d ∈ terminal_dims, dims_out)
        T_inds[dout] = sublinkdims[dout]                          # end in channel linkdims[d] for each dimension d
        @assert isone(T_inds[dout])
        block_helper_inds[dout] = nblocks(linkinds[dout])
      end

      # set non-trivial helper inds
      for d in normal_dims
        block_helper_inds[d] = qnblock(linkinds[d], T_qns[d])
      end
      @assert all(≠(-1), block_helper_inds)# check that all block indices are set

      # make and fill Block 
      theblock = Block(Tuple(block_helper_inds))
      if isempty(normal_dims)
        M = get!(blocks, (theblock, terms(t)), zero_arr())
        @assert isone(length(M))
        M[] += ct
      else
        M = get!(blocks, (theblock, terms(t)), zero_arr())
        dim_ranges = Tuple(size(Vv[d][T_qns[d]], 2) for d in normal_dims)
        for c in CartesianIndices(dim_ranges) # applies isometries in a element-wise manner
          z = ct
          temp_inds = copy(T_inds)
          for (i, d) in enumerate(normal_dims)
            V_factor = Vv[d][T_qns[d]][T_inds[d], c[i]]
            z *= (d == dim_in ? conj(V_factor) : V_factor) # conjugate incoming isometry factor
            temp_inds[d] = c[i]
          end
          M[temp_inds...] += z
        end
      end
    end

    H[v] = ITensor()

    # Set the final arrow directions
    if !isnothing(dim_in)
      linkinds[dim_in] = dag(linkinds[dim_in])
    end

    for ((b, q_op), m) in blocks
      Op = computeSiteProd(sites, Prod(q_op))
      if hasqns(Op)
        # FIXME: this may not be safe, we may want to check for the equivalent (zero tensor?) case in the dense case as well
        iszero(nnzblocks(Op)) && continue
      end
      sq = flux(Op)
      if !isnothing(sq)
        rq = (b[1] == 1 ? Hflux : first(space(linkinds[1])[b[1]])) # get row (dim_in) QN
        cq = rq - sq # get column (out_dims) QN
        if ITensors.using_auto_fermion()
          # we need to account for the direct product below ordering the physical indices as the last indices
          # although they are in between incoming and outgoing indices in the canonical site-ordering
          perm = (1, 3, 2)
          if ITensors.compute_permfactor(perm, rq, sq, cq) == -1
            Op .*= -1
          end
        end
      end
      T = ITensors.BlockSparseTensor(coefficient_type, [b], linkinds)
      T[b] .= m
      iT = itensor(T)
      if !thishasqns
        iT = removeqns(iT)
      end

      if is_internal[v]
        H[v] += iT
      else
        #TODO: Remove this assert since it seems to be costly
        #if hasqns(iT)
        #  @assert flux(iT * Op) == Hflux
        #end
        H[v] += (iT * Op)
      end
    end

    linkdims = dim.(linkinds)
    # add starting and ending identity operators
    idT = zeros(coefficient_type, linkdims...)
    if isnothing(dim_in)
      # only one real starting identity
      idT[ones(Int, v_degree)...] = 1.0
    end
    # ending identities are a little more involved
    if !isnothing(dim_in)
      # place identity if all channels end
      idT[linkdims...] = 1.0
      # place identity from start of incoming channel to start of each single outgoing channel, and end all other channels
      idT_end_inds = [linkdims...]
      #this should really be an int
      idT_end_inds[dim_in] = 1
      for dout in dims_out
        idT_end_inds[dout] = 1
        idT[idT_end_inds...] = 1.0
        # reset
        idT_end_inds[dout] = linkdims[dout]
      end
    end

    T = itensor(idT, linkinds)
    if !thishasqns
      T = removeqns(T)
    end
    if is_internal[v]
      H[v] += T
    else
      H[v] += T * ITensorNetworks.computeSiteProd(sites, Prod([(Op("Id", v))]))
    end
  end
  return H
end

#
# TermQNMap implements a function with an internal cache.
# Calling it on a term determines that term's flux
# but tries not to rebuild the corresponding ITensor.
# (Previously it was the function calc_qn.)
#
struct TermQNMap{V}
  sites::IndsNetwork
  op_cache::Dict{Pair{String,V},ITensor}
  TermQNMap{V}(s) where {V} = new{V}(s, Dict{Pair{String,V},ITensor}())
end

function (t::TermQNMap)(term)
  q = QN()
  for st in term
    op_tensor = get(t.op_cache, which_op(st) => site(st), nothing)
    if op_tensor === nothing
      op_tensor = op(t.sites[site(st)], which_op(st); params(st)...)
      t.op_cache[which_op(st) => site(st)] = op_tensor
    end
    if !isnothing(flux(op_tensor))
      q += flux(op_tensor)
    end
  end
  return q
end

function ttn_svd(
  coefficient_type::Type{<:Number}, os::OpSum, sites::IndsNetwork, root_vertex; kws...
)
  term_qn_map = TermQNMap{vertextype(sites)}(sites)

  # Traverse tree outwards from root vertex
  ordered_verts = _default_vertex_ordering(sites, root_vertex)
  # Store edges in fixed ordering relative to root
  ordered_edges = _default_edge_ordering(sites, root_vertex)

  symbolic_ttn, inbond_coefs = make_symbolic_ttn(
    coefficient_type, os, sites; ordered_verts, ordered_edges, root_vertex, term_qn_map
  )

  Vs = svd_bond_coefs(
    coefficient_type, sites, inbond_coefs; ordered_verts, ordered_edges, kws...
  )

  Hflux = -term_qn_map(terms(first(terms(os))))

  T = compress_ttn(
    coefficient_type, sites, Hflux, symbolic_ttn, Vs; ordered_verts, ordered_edges
  )

  return T
end

# 
# Tree adaptations of functionalities in ITensors.jl/src/physics/autompo/opsum_to_mpo_generic.jl
# 

# TODO: fix fermion support, definitely broken

# needed an extra `only` compared to ITensors version since IndsNetwork has Vector{<:Index}
# as vertex data
function isfermionic(t::Vector{Op}, sites::IndsNetwork{V,<:Index}) where {V}
  p = +1
  for op in t
    if has_fermion_string(name(op), only(sites[site(op)]))
      p *= -1
    end
  end
  return (p == -1)
end

# only(site(ops[1])) in ITensors breaks for Tuple site labels, had to drop the only
function computeSiteProd(sites::IndsNetwork{V,<:Index}, ops::Prod{Op})::ITensor where {V}
  v = site(ops[1])
  T = op(sites[v], which_op(ops[1]); params(ops[1])...)
  for j in 2:length(ops)
    (site(ops[j]) != v) && error("Mismatch of vertex labels in computeSiteProd")
    opj = op(sites[v], which_op(ops[j]); params(ops[j])...)
    T = product(T, opj)
  end
  return T
end

function _default_vertex_ordering(g::AbstractGraph, root_vertex)
  return reverse(post_order_dfs_vertices(g, root_vertex))
end

function _default_edge_ordering(g::AbstractGraph, root_vertex)
  return reverse(reverse.(post_order_dfs_edges(g, root_vertex)))
end

function check_terms_support(os::OpSum, sites)
  for t in os
    if !all(map(v -> has_vertex(sites, v), ITensors.sites(t)))
      error(
        "The OpSum contains a term $t that does not have support on the underlying graph."
      )
    end
  end
end

# This code is very similar to ITensorMPS sorteachterm in opsum_generic.jl
function sorteachterm(os::OpSum, sites, root_vertex)
  os = copy(os)

  # Build the isless_site function to pass to sortperm below:
  # + ordering = array of vertices ordered relative to chosen root, chosen outward from root
  # + site_positions = map from vertex to where it is in ordering (inverse map of `ordering`)
  ordering = _default_vertex_ordering(sites, root_vertex)
  site_positions = Dict(zip(ordering, 1:length(ordering)))
  isless_site(o1::Op, o2::Op) = site_positions[site(o1)] < site_positions[site(o2)]

  N = nv(sites)
  for j in eachindex(os)
    t = os[j]

    # Sort operators in t by site order,
    # and keep the permutation used, perm, for analysis below
    Nt = length(t)
    #perm = Vector{Int}(undef, Nt)
    perm = sortperm(terms(t); alg=InsertionSort, lt=isless_site)
    t = coefficient(t) * Prod(terms(t)[perm])

    # Everything below deals with fermionic operators:

    # Identify fermionic operators,
    # zeroing perm for bosonic operators,
    # and inserting string "F" operators
    prevsite = typemax(Int) #keep track of whether we are switching to a new site
    t_parity = +1
    for n in reverse(1:Nt)
      currsite = site(t[n])
      fermionic = has_fermion_string(which_op(t[n]), only(sites[site(t[n])]))
      if !ITensors.using_auto_fermion() && (t_parity == -1) && (currsite < prevsite)
        error("No verified fermion support for automatic TTN constructor!") # no verified support, just throw error
        # Put local piece of Jordan-Wigner string emanating
        # from fermionic operators to the right
        # (Remaining F operators will be put in by svdMPO)
        terms(t)[n] = Op("$(which_op(t[n])) * F", only(site(t[n])))
      end
      prevsite = currsite

      if fermionic
        t_parity = -t_parity
      else
        # Ignore bosonic operators in perm
        # by zeroing corresponding entries
        perm[n] = 0
      end
    end
    if t_parity == -1
      error("Parity-odd fermionic terms not yet supported by AutoTTN")
    end

    # Keep only fermionic op positions (non-zero entries)
    filter!(!iszero, perm)
    # and account for anti-commuting, fermionic operators 
    # during above sort; put resulting sign into coef
    t *= ITensors.parity_sign(perm)
    terms(os)[j] = t
  end
  return os
end

function sortmergeterms(os::OpSum{C}) where {C}
  os_sorted_terms = sort(terms(os))
  os = Sum(os_sorted_terms)
  # Merge (add) terms with same operators
  merge_os_data = Scaled{C,Prod{Op}}[]
  last_term = copy(os[1])
  last_term_coef = coefficient(last_term)
  for n in 2:length(os)
    if argument(os[n]) == argument(last_term)
      last_term_coef += coefficient(os[n])
      last_term = last_term_coef * argument(last_term)
    else
      push!(merge_os_data, last_term)
      last_term = os[n]
      last_term_coef = coefficient(last_term)
    end
  end
  push!(merge_os_data, last_term)
  os = Sum(merge_os_data)
  return os
end

"""
    ttn(os::OpSum, sites::IndsNetwork{<:Index}; kwargs...)
    ttn(eltype::Type{<:Number}, os::OpSum, sites::IndsNetwork{<:Index}; kwargs...)
       
Convert an OpSum object `os` to a TreeTensorNetwork, with indices given by `sites`.
"""
function ttn(
  os::OpSum,
  sites::IndsNetwork;
  root_vertex=GraphsExtensions.default_root_vertex(sites),
  kwargs...,
)
  length(terms(os)) == 0 && error("OpSum has no terms")
  is_tree(sites) || error("Site index graph must be a tree.")
  is_leaf_vertex(sites, root_vertex) || error("Tree root must be a leaf vertex.")
  check_terms_support(os, sites)
  os = deepcopy(os) #TODO: do we need this? sorteachterm copies `os` again
  os = sorteachterm(os, sites, root_vertex)
  os = sortmergeterms(os)
  return ttn_svd(os, sites, root_vertex; kwargs...)
end

function mpo(os::OpSum, external_inds::Vector; kwargs...)
  return ttn(os, path_indsnetwork(external_inds); kwargs...)
end
function mpo(os::OpSum, s::IndsNetwork; kwargs...)
  # TODO: Check it is a path graph.
  return ttn(os, s; kwargs...)
end

# Conversion from other formats
function ttn(o::Op, s::IndsNetwork; kwargs...)
  return ttn(OpSum{Float64}() + o, s; kwargs...)
end

function ttn(o::Scaled{C,Op}, s::IndsNetwork; kwargs...) where {C}
  return ttn(OpSum{C}() + o, s; kwargs...)
end

function ttn(o::Sum{Op}, s::IndsNetwork; kwargs...)
  return ttn(OpSum{Float64}() + o, s; kwargs...)
end

function ttn(o::Prod{Op}, s::IndsNetwork; kwargs...)
  return ttn(OpSum{Float64}() + o, s; kwargs...)
end

function ttn(o::Scaled{C,Prod{Op}}, s::IndsNetwork; kwargs...) where {C}
  return ttn(OpSum{C}() + o, s; kwargs...)
end

function ttn(o::Sum{Scaled{C,Op}}, s::IndsNetwork; kwargs...) where {C}
  return ttn(OpSum{C}() + o, s; kwargs...)
end

# Catch-all for leaf eltype specification
function ttn(eltype::Type{<:Number}, os, sites::IndsNetwork; kwargs...)
  return NDTensors.convert_scalartype(eltype, ttn(os, sites; kwargs...))
end

# 
# Sparse finite state machine construction
# 

# Allow sparse arrays with ITensors.Sum entries
function Base.zero(::Type{S}) where {S<:Sum}
  return S()
end
Base.zero(t::Sum) = zero(typeof(t))
