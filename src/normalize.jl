using LinearAlgebra

function rescale(tn::AbstractITensorNetwork; alg="exact", kwargs...)
  return rescale(Algorithm(alg), tn; kwargs...)
end

function rescale(
  alg::Algorithm"exact",
  tn::AbstractITensorNetwork;
  vs_to_rescale=collect(vertices(tn)),
  kwargs...,
)
  logn = logscalar(alg, tn; kwargs...)
  c = 1.0 / (exp(logn / length(vs_to_rescale)))
  tn = copy(tn)
  for v in vs_to_rescale
    tn[v] *= c
  end
  return tn
end

function rescale(
  alg::Algorithm,
  tn::AbstractITensorNetwork;
  vs_to_rescale=collect(vertices(tn)),
  (cache!)=nothing,
  cache_construction_kwargs=default_cache_construction_kwargs(alg, tn),
  update_cache=isnothing(cache!),
  cache_update_kwargs=default_cache_update_kwargs(alg),
)
  if isnothing(cache!)
    cache! = Ref(cache(alg, tn; cache_construction_kwargs...))
  end

  if update_cache
    cache![] = update(cache![]; cache_update_kwargs...)
  end

  cache![] = rescale(cache![]; vs_to_rescale)

  return tensornetwork(cache![])
end

function LinearAlgebra.normalize(tn::AbstractITensorNetwork; alg="exact", kwargs...)
  return normalize(Algorithm(alg), tn; kwargs...)
end

function LinearAlgebra.normalize(
  alg::Algorithm"exact", tn::AbstractITensorNetwork; kwargs...
)
  norm_tn = inner_network(tn, tn)
  return ket_network(rescale(alg, norm_tn; kwargs...))
end

function LinearAlgebra.normalize(
  alg::Algorithm,
  tn::AbstractITensorNetwork;
  (cache!)=nothing,
  cache_construction_function=tn ->
    cache(alg, tn; default_cache_construction_kwargs(alg, tn)...),
  update_cache=isnothing(cache!),
  cache_update_kwargs=default_cache_update_kwargs(alg),
  cache_construction_kwargs=(;),
)
  norm_tn = inner_network(tn, tn)
  if isnothing(cache!)
    cache! = Ref(cache(alg, norm_tn; cache_construction_kwargs...))
  end

  vs = collect(vertices(tn))
  vs_to_rescale = vcat(
    [ket_vertex(norm_tn, v) for v in vs], [bra_vertex(norm_tn, v) for v in vs]
  )
  norm_tn = rescale(alg, norm_tn; vs_to_rescale, cache!, update_cache, cache_update_kwargs)

  return ket_network(norm_tn)
end
