using ITensors: siteinds, Op, prime, OpSum, apply, Trotter
using Graphs: AbstractGraph, SimpleGraph, edges, vertices, is_tree, connected_components
using NamedGraphs: NamedGraph, NamedEdge, NamedGraphs, rename_vertices
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using NamedGraphs.GraphsExtensions:
  decorate_graph_edges,
  forest_cover,
  add_edges,
  rem_edges,
  add_vertices,
  rem_vertices,
  disjoint_union,
  subgraph,
  src,
  dst,
  degree
using NamedGraphs.PartitionedGraphs: PartitionVertex, partitionedge, unpartitioned_graph
using ITensorNetworks:
  BeliefPropagationCache,
  AbstractITensorNetwork,
  AbstractFormNetwork,
  IndsNetwork,
  ITensorNetwork,
  insert_linkinds,
  ttn,
  union_all_inds,
  neighbor_vertices,
  environment,
  messages,
  update_factor,
  message,
  partitioned_tensornetwork,
  bra_vertex,
  ket_vertex,
  operator_vertex,
  default_cache_update_kwargs,
  dual_index_map,
  norm_sqr_network
using DataGraphs: underlying_graph
using ITensorNetworks.ModelHamiltonians: heisenberg
using ITensors:
  ITensor,
  noprime,
  dag,
  noncommonind,
  commonind,
  replaceind,
  dim,
  noncommoninds,
  delta,
  replaceinds,
  dense
using ITensors.NDTensors: denseblocks
using Dictionaries: set!
using SplitApplyCombine: group

function BP_apply(
  o::ITensor, ψ::AbstractITensorNetwork, bpc::BeliefPropagationCache; reset_all_messages = false, apply_kwargs...
)
  bpc = copy(bpc)
  ψ = copy(ψ)
  vs = neighbor_vertices(ψ, o)
  envs = environment(bpc, PartitionVertex.(vs))
  singular_values! = Ref(ITensor())
  ψ = noprime(apply(o, ψ; envs, singular_values!, normalize=true, apply_kwargs...))
  ψdag = prime(dag(ψ); sites=[])
  if length(vs) == 2
    v1, v2 = vs
    pe = partitionedge(bpc, (v1, "bra") => (v2, "bra"))
    mts = messages(bpc)
    ind2 = commonind(singular_values![], ψ[v1])
    δuv = dag(copy(singular_values![]))
    δuv = replaceind(δuv, ind2, ind2')
    map_diag!(sign, δuv, δuv)
    singular_values![] = denseblocks(singular_values![]) * denseblocks(δuv)
    if !reset_all_messages
        set!(mts, pe, dag.(ITensor[singular_values![]]))
        set!(mts, reverse(pe), ITensor[singular_values![]])
    else
        bpc = BeliefPropagationCache(partitioned_tensornetwork(bpc))
    end
end
  for v in vs
    bpc = update_factor(bpc, (v, "ket"), ψ[v])
    bpc = update_factor(bpc, (v, "bra"), ψdag[v])
  end
  return ψ, bpc
end

function smallest_eigvalue(A::AbstractITensorNetwork)
  out = reduce(*, [A[v] for v in vertices(A)])
  out = out * combiner(inds(out; plev=0)) * combiner(inds(out; plev=1))
  out = array(out)
  return minimum(real.(eigvals(out)))
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

function contract_heavy_hex_state(ψ::ITensorNetwork)
  ψ = copy(ψ)
  degree_two_vertices = filter(v -> degree(ψ, v) == 2, vertices(ψ))
  for v in degree_two_vertices
    vn = first(neighbors(ψ, v))
    ψ = contract(ψ, NamedEdge(v => vn), merged_vertex = vn)
  end

  return contract([dense(ψ[v]) for v in vertices(ψ)]; sequence = "automatic")
end

function exact_heavy_hex_energy(ψ::ITensorNetwork, H::OpSum)
  s = indsnetwork(ψ)
  ψ_sv = dense(contract_heavy_hex_state(ψ))
  z = (ψ_sv * dag(ψ_sv))[]
  out = 0
  op_tensors = Vector{ITensor}(H, s)
  for op_tensor in op_tensors
    ψo_sv = noprime(dense(op_tensor)*ψ_sv)
    out += (ψo_sv * dag(ψ_sv))[]
  end

  return out / z
end

function opsum_to_edge_dict(s::IndsNetwork, H::OpSum)
  es = edges(s)
  op_tensors = Vector{ITensor}(H, s)
  term_dict = Dictionary(edges(s), [ITensor() for e in edges(s)])
  for (i, term) in enumerate(H)
    if length(sites(term)) == 1 && !iszero(first(term.args))
      v = only(sites(term))
      e = first(filter(e -> src(e) == v || dst(e) == v, edges(s)))
      vother = src(e) == v ? dst(e) : src(e)
      set!(term_dict, e, term_dict[e] + op_tensors[i] * delta(s[vother], s[vother]'))
    elseif length(sites(term)) == 2 && !iszero(first(term.args))
      v1, v2 = first(sites(term)), last(sites(term))
      ed = NamedEdge(v1 => v2)
      actual_ed = ed ∈ edges(s) ? ed : reverse(ed)
      set!(term_dict, actual_ed, term_dict[actual_ed] + op_tensors[i])
    end
  end

  new_term_dict = Dictionary()
  for e in keys(term_dict)
    v1, v2 = src(e), dst(e)
    Ov1, Ov2 = factorize_svd(term_dict[e], s[v1], s[v1]'; cutoff = 1e-16)
    set!(new_term_dict, e, [Ov1, Ov2])
  end

  return new_term_dict
end

#Construct a graph with edges everywhere a two-site gate appears.
function build_graph_from_interactions(list)
  vertices = []
  edges = []
  for term in list
      vsrc, vdst = (term[3],), (term[4],)
      if vsrc ∉ vertices
          push!(vertices, vsrc)
      end
      if vdst ∉ vertices
          push!(vertices, vdst)
      end
      e = NamedEdge(vsrc => vdst)
      if e ∉ edges || reverse(e) ∉ edges
          push!(edges, e)
      end
  end
  g = NamedGraph()
  g = add_vertices(g, vertices)
  g = add_edges(g, edges)
  return g
end

function graph_to_adj_mat(g::AbstractGraph)
  L = length(vertices(g))
  adj_mat = zeros((L, L))
  for (i, v1) in enumerate(vertices(g))
      for (j, v2) in enumerate(vertices(g))
          if NamedEdge(v1 => v2) ∈ edges(g) || NamedEdge(v2 => v1) ∈ edges(g)
              adj_mat[i, j] = 1
          end
      end
  end
  return adj_mat
end

function ising_adjmat(L, adj_mat; J, h, hl)
  os = OpSum()
  for i in 1:L
      for j in (i+1):L
          if !iszero(J * adj_mat[i, j])
              os += J * adj_mat[i, j], "Sz", i, "Sz", j
          end
      end
  end
  for i in 1:L
    os += h, "Sx", i
    os += hl, "Sz", i
  end

  return os
end

function heisenberg_adjmat(L, adj_mat; J = 1, Δ=1)
  os = OpSum()
  for i in 1:L
      for j in (i+1):L
          if !iszero(J * adj_mat[i, j])
              os += J * adj_mat[i, j] / 2, "S+", i, "S-", j
              os += J * adj_mat[i, j] / 2, "S-", i, "S+", j
              os += J * adj_mat[i, j] * Δ, "Sz", i, "Sz", j
          end
      end
  end
  return os
end

function xyz_adjmat(L, adj_mat; Jx = 1, Jy = 1, Jz = 1)
  os = OpSum()
  for i in 1:L
      for j in (i+1):L
          if !iszero(Jx * adj_mat[i, j])
              os += Jx * adj_mat[i, j], "Sx", i, "Sx", j
          end
          if !iszero(Jy * adj_mat[i, j])
            os += Jy * adj_mat[i, j], "Sy", i, "Sy", j
          end
          if !iszero(Jz * adj_mat[i, j])
            os += Jz * adj_mat[i, j], "Sz", i, "Sz", j
          end
      end
  end
  return os
end

function imaginary_time_evo(
  s::IndsNetwork,
  ψ::ITensorNetwork,
  model::Function,
  dbetas::Vector{<:Tuple};
  model_params,
  bp_update_kwargs=(; maxiter=10, tol=1e-10),
  apply_kwargs=(; cutoff=1e-12, maxdim=10),
)
  ψ = copy(ψ)
  g = underlying_graph(ψ)
  L = length(vertices(g))

  ℋ = filter_zero_terms(model(g; model_params...))
  ψψ = norm_sqr_network(ψ)
  bpc = BeliefPropagationCache(ψψ, group(v -> v[1], vertices(ψψ)))
  bpc = update(bpc; bp_update_kwargs...)
  println("Starting Imaginary Time Evolution")
  β = 0
  for (i, period) in enumerate(dbetas)
    nbetas, dβ = first(period), last(period)
    println("Entering evolution period $i , β = $β, dβ = $dβ")
    U = exp(-dβ * ℋ; alg=Trotter{2}())
    gates = Vector{ITensor}(U, s)
    for i in 1:nbetas
      for gate in gates
        ψ, bpc = BP_apply(gate, ψ, bpc; apply_kwargs...)
      end
      β += dβ
      bpc = update(bpc; bp_update_kwargs...)
    end
    e = sum(expect(ψ, ℋ; alg="bp"))
    Z = sum(expect(ψ, "Z"; alg = "bp"))
    println("Total Mag is $(Z/L)")
    println("Energy is $(e/L)")
  end

  return ψ
end

function filter_zero_terms(H::OpSum)
  new_H = OpSum()
  for h in H
    if !iszero(first(h.args))
      new_H += h
    end
  end
  return new_H
end
