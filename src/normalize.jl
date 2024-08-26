using LinearAlgebra

function LinearAlgebra.normalize(tn::AbstractITensorNetwork; alg="exact", kwargs...)
  return normalize(Algorithm(alg), tn; kwargs...)
end

function LinearAlgebra.normalize(alg::Algorithm"exact", tn::AbstractITensorNetwork)
  norm_tn = norm_sqr_network(tn)
  log_norm = logscalar(alg, norm_tn)
  tn = copy(tn)
  L = length(vertices(tn))
  c = exp(log_norm / L)
  for v in vertices(tn)
    tn[v] = tn[v] / sqrt(c)
  end
  return tn
end

function LinearAlgebra.normalize(
  alg::Algorithm"bp",
  tn::AbstractITensorNetwork;
  (cache!)=nothing,
  update_cache=isnothing(cache!),
  cache_update_kwargs=default_cache_update_kwargs(cache!),
)
  if isnothing(cache!)
    tntn = norm_sqr_network(tn)
    cache! = Ref(BeliefPropagationCache(tntn, group(v -> first(v), vertices(tntn))))
  end

  if update_cache
    cache![] = update(cache![]; cache_update_kwargs...)
  end

  tn = copy(tn)
  cache![] = normalize_messages(cache![])
  norm_tn = tensornetwork(cache![])

  vertices_states = Dictionary()
  for v in vertices(tn)
    v_ket, v_bra = (v, "bra"), (v, "ket")
    pv = only(partitionvertices(cache![], [v_ket]))
    vn = region_scalar(cache![], pv)
    state = norm_tn[v_ket] / sqrt(vn)
    state_dag = norm_tn[v_bra]
    set!(vertices_states, v_ket, state)
    set!(vertices_states, v_bra, state_dag)
    tn[v] = state
  end

  cache![] = update_factors(cache![], vertices_states)

  return tn
end