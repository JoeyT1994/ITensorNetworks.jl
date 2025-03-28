using Dictionaries: Dictionary, set!
using ITensors: Op, op, contract, siteinds, which_op
using ITensorMPS: ITensorMPS, expect

default_expect_alg() = "bp"

function ITensorMPS.expect(ψIψ::AbstractFormNetwork, op::Op; kwargs...)
  v = only(op.sites)
  ψIψ_v = ψIψ[operator_vertex(ψIψ, v)]
  s = commonind(ψIψ[ket_vertex(ψIψ, v)], ψIψ_v)
  operator = ITensors.op(op.which_op, s)
  ∂ψIψ_∂v = environment(ψIψ, operator_vertices(ψIψ, [v]); kwargs...)
  numerator_ts = vcat(∂ψIψ_∂v, operator)
  denominator_ts = vcat(∂ψIψ_∂v, ψIψ_v)
  numerator_seq = contraction_sequence(numerator_ts; alg="optimal")
  denominator_seq = contraction_sequence(denominator_ts; alg="optimal")
  numerator = contract(numerator_ts; sequence)[]
  denominator = contract(denominator_ts; sequence)[]

  return numerator / denominator
end

function ITensorMPS.expect(
  alg::Algorithm,
  ψ::AbstractITensorNetwork,
  ops;
  (cache!)=nothing,
  update_cache=isnothing(cache!),
  cache_update_kwargs=default_cache_update_kwargs(alg),
  cache_construction_kwargs=default_cache_construction_kwargs(alg, inner_network(ψ, ψ)),
  kwargs...,
)
  ψIψ = inner_network(ψ, ψ)
  if isnothing(cache!)
    cache! = Ref(cache(alg, ψIψ; cache_construction_kwargs...))
  end

  if update_cache
    cache![] = update(cache![]; cache_update_kwargs...)
  end

  return map(op -> expect(ψIψ, op; alg, cache!, update_cache=false, kwargs...), ops)
end

function ITensorMPS.expect(alg::Algorithm"exact", ψ::AbstractITensorNetwork, ops; kwargs...)
  ψIψ = inner_network(ψ, ψ)
  return map(op -> expect(ψIψ, op; alg, kwargs...), ops)
end

function ITensorMPS.expect(
  ψ::AbstractITensorNetwork, op::Op; alg=default_expect_alg(), kwargs...
)
  return expect(Algorithm(alg), ψ, [op]; kwargs...)
end

function ITensorMPS.expect(
  ψ::AbstractITensorNetwork, op::String, vertices; alg=default_expect_alg(), kwargs...
)
  return expect(Algorithm(alg), ψ, [Op(op, vertex) for vertex in vertices]; kwargs...)
end

function ITensorMPS.expect(
  ψ::AbstractITensorNetwork, op::String; alg=default_expect_alg(), kwargs...
)
  return expect(ψ, op, vertices(ψ); alg, kwargs...)
end
