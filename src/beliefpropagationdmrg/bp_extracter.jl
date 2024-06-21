using ITensors: scalartype, which_op, name, names, sites, OpSum
using ITensorNetworks:
  AbstractITensorNetwork, ket_vertices, bra_vertices, tensornetwork, default_message_update, operator_network
using ITensorNetworks.ITensorsExtensions: map_eigvals
using NamedGraphs.GraphsExtensions: a_star, neighbors, boundary_edges

include("treetensornetworkoperators.jl")

function effective_environments(state::ITensorNetwork, H::OpSum, ψIψ_bpc::BeliefPropagationCache, region)
  environments = Vector{ITensor}[]
  s = siteinds(state)
  op_tensors = Vector{ITensor}(H, s)
  for (i, term) in enumerate(H)
    term_envs = ITensor[]
    if length(sites(term)) == 1 && !iszero(first(term.args))
      paths = [a_star(state, only(sites(term)), v) for v in region]
      path = first(sort(paths; by = length))
      ψOψ_qf = QuadraticFormNetwork(state)
      ψOψ_qf[(only(sites(term)), "operator")] = op_tensors[i]
    elseif length(sites(term)) == 2 && !iszero(first(term.args))
      v1, v2 = first(sites(term)), last(sites(term))
      if v1 ∈ region && v2 ∈ region
        path = []
      elseif v1 ∈ region && v2 ∉ region
        path = NamedEdge[NamedEdge(v2 => v1)]
      elseif v2 ∈ region && v1 ∉ region
        path = NamedEdge[NamedEdge(v1 => v2)]
      else
        paths = [a_star(state, v1, v) for v in region]
        path = first(sort(paths; by = length))
        if v2 ∉ src.(path) && v2 ∉ dst.(path)
          prepend!(path, NamedEdge[NamedEdge(v2 => v1)])
        end
      end
      Ov1, Ov2 = factorize_svd(op_tensors[i], s[v1], s[v1]'; cutoff = 1e-16)
      ψOψ_qf = QuadraticFormNetwork(state)
      ψOψ_qf[(v1, "operator")] = Ov1
      ψOψ_qf[(v2, "operator")] = Ov2
    else 
      path = nothing
    end
    if !isnothing(path)
      env = ITensor(1.0)
      path = vcat(src.(path))
      for v in path
        env *= ψOψ_qf[(v, "bra")]* ψOψ_qf[(v, "operator")] * ψOψ_qf[(v, "ket")]
        vns = neighbors(state, v)
        for vn in vns
          if vn ∉ path && vn ∉ region
            env *= only(message(ψIψ_bpc, PartitionEdge(vn => v)))
          end
        end
      end
      push!(term_envs, env)
      for e in boundary_edges(state, region; dir = :in)
        if src(e) ∉ path
          push!(term_envs, only(message(ψIψ_bpc, PartitionEdge(e))))
        end
      end
      append!(term_envs, ITensor[ψOψ_qf[(v, "operator")] for v in region])

      push!(environments, term_envs)
    end
  end

  return environments
end

function effective_environments(state::ITensorNetwork, term_dict, ψIψ_bpc::BeliefPropagationCache, region)
  environments = Vector{ITensor}[]
  s = siteinds(state)
  for e in keys(term_dict)
    term_envs = ITensor[]
    v1, v2 = src(e), dst(e)
    if v1 ∈ region && v2 ∈ region
      path = []
    elseif v1 ∈ region && v2 ∉ region
      path = NamedEdge[NamedEdge(v2 => v1)]
    elseif v2 ∈ region && v1 ∉ region
      path = NamedEdge[NamedEdge(v1 => v2)]
    else
      paths = [a_star(state, v1, v) for v in region]
      path = first(sort(paths; by = length))
      if v2 ∉ src.(path) && v2 ∉ dst.(path)
        prepend!(path, NamedEdge[NamedEdge(v2 => v1)])
      end
    end
    ψOψ_qf = QuadraticFormNetwork(state)
    ψOψ_qf[(v1, "operator")] = first(term_dict[e])
    ψOψ_qf[(v2, "operator")] = last(term_dict[e])
    if !isnothing(path)
      env = ITensor(1.0)
      path = vcat(src.(path))
      for v in path
        env *= ψOψ_qf[(v, "bra")]* ψOψ_qf[(v, "operator")] * ψOψ_qf[(v, "ket")]
        vns = neighbors(state, v)
        for vn in vns
          if vn ∉ path && vn ∉ region
            env *= only(message(ψIψ_bpc, PartitionEdge(vn => v)))
          end
        end
      end
      push!(term_envs, env)
      for e in boundary_edges(state, region; dir = :in)
        if src(e) ∉ path
          push!(term_envs, only(message(ψIψ_bpc, PartitionEdge(e))))
        end
      end
      append!(term_envs, ITensor[ψOψ_qf[(v, "operator")] for v in region])

      push!(environments, term_envs)
    end
  end

  return environments
end

function effective_environments_V2(state::ITensorNetwork, H::OpSum, ψIψ_bpc::BeliefPropagationCache, region)
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

function bp_extracter(
  ψ::AbstractITensorNetwork,
  operators::Vector{<:ITensorNetwork},
  ψIψ_bpc::BeliefPropagationCache,
  region;
  regularization=10 * eps(scalartype(ψ)),
  ishermitian=true,
)

  form_network = tensornetwork(ψIψ_bpc)
  form_ket_vertices, form_bra_vertices = ket_vertices(form_network, region),
  bra_vertices(form_network, region)

  ∂ψOψ_bpc_∂rs = effective_environments(ψ, operators, ψIψ_bpc, region)
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

  ∂ψOψ_bpc_∂rs = effective_environments_V2(ψ, H, ψIψ_bpc, region)
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

function bp_extracter(
  ψ::AbstractITensorNetwork,
  term_dict,
  ψIψ_bpc::BeliefPropagationCache,
  region;
  regularization=0,
  ishermitian=true,
)

  form_network = tensornetwork(ψIψ_bpc)
  form_ket_vertices, form_bra_vertices = ket_vertices(form_network, region),
  bra_vertices(form_network, region)

  ∂ψOψ_bpc_∂rs = effective_environments(ψ, term_dict, ψIψ_bpc, region)
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

