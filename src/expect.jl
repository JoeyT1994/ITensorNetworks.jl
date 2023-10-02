function expect(
  op::String,
  ψ::AbstractITensorNetwork;
  cutoff=nothing,
  maxdim=nothing,
  ortho=false,
  sequence=nothing,
  expec_vertices=vertices(ψ),
)
  s = siteinds(ψ)
  ElT = promote_itensor_eltype(ψ)
  # ElT = ishermitian(ITensors.op(op, s[vertices[1]])) ? real(ElT) : ElT
  res = Dictionary(expec_vertices, Vector{ElT}(undef, length(expec_vertices)))
  if isnothing(sequence)
    sequence = contraction_sequence(inner_network(ψ, ψ; flatten=true))
  end
  normψ² = norm_sqr(ψ; sequence)
  for v in expec_vertices
    O = ITensor(Op(op, v), s)
    Oψ = apply(O, ψ; cutoff, maxdim, ortho)
    res[v] = contract_inner(ψ, Oψ; sequence) / normψ²
  end
  return res
end

function expect(
  ℋ::OpSum,
  ψ::AbstractITensorNetwork;
  cutoff=nothing,
  maxdim=nothing,
  ortho=false,
  sequence=nothing,
)
  s = siteinds(ψ)
  # h⃗ = Vector{ITensor}(ℋ, s)
  if isnothing(sequence)
    sequence = contraction_sequence(inner_network(ψ, ψ; flatten=true))
  end
  h⃗ψ = [apply(hᵢ, ψ; cutoff, maxdim, ortho) for hᵢ in ITensors.terms(ℋ)]
  ψhᵢψ = [contract_inner(ψ, hᵢψ; sequence) for hᵢψ in h⃗ψ]
  ψh⃗ψ = sum(ψhᵢψ)
  ψψ = norm_sqr(ψ; sequence)
  return ψh⃗ψ / ψψ
end

function expect(
  opsum_sum::Sum{<:OpSum},
  ψ::AbstractITensorNetwork;
  cutoff=nothing,
  maxdim=nothing,
  ortho=true,
  sequence=nothing,
)
  return expect(sum(Ops.terms(opsum_sum)), ψ; cutoff, maxdim, ortho, sequence)
end

function expect_BP(
  op::String,
  ψ::AbstractITensorNetwork,
  ψψ::AbstractITensorNetwork,
  mts::DataGraph;
  expect_vertices=vertices(ψ),
)
  s = siteinds(ψ)
  ElT = promote_itensor_eltype(ψ)
  res = Dictionary(expect_vertices, Vector{ElT}(undef, length(expect_vertices)))
  for v in expect_vertices
    O = ITensor(Op(op, v), s)
    numerator_network = approx_network_region(
      ψψ, mts, [(v, 1)]; verts_tn=ITensorNetwork(ITensor[apply(O, ψ[v])])
    )
    denominator_network = approx_network_region(ψψ, mts, [(v, 1)])
    res[v] = contract(numerator_network)[] / contract(denominator_network)[]
  end

  return res
end

function expect_BP(op::String, ψ::AbstractITensorNetwork; expect_vertices=vertices(ψ))
  ψψ = norm_network(ψ)
  mts = belief_propagation(
    ψψ,
    message_tensors(ψψ; subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ)))));
    contract_kwargs=(; alg="exact"),
  )
  return expect_BP(op, ψ, ψψ, mts; expect_vertices)
end

function expect_BP(
  op::String,
  ψ::AbstractITensorNetwork,
  ψψ::AbstractITensorNetwork;
  expect_vertices=vertices(ψ),
)
  mts = belief_propagation(
    ψψ,
    message_tensors(ψψ; subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ)))));
    contract_kwargs=(; alg="exact"),
  )
  return expect_BP(op, ψ, ψψ, mts; expect_vertices)
end
