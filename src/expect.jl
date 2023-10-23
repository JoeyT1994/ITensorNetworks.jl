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
  op::Vector{String},
  ψ::AbstractITensorNetwork,
  ψψ::AbstractITensorNetwork,
  mts::DataGraph,
  vertex_set;
  sequence = nothing)

  s = siteinds(ψ)
  @assert length(vertex_set) == length(op)
  numerator_network = approx_network_region(
      ψψ, mts, [(v, 1) for v in vertex_set]; verts_tn=ITensorNetwork(ITensor[apply(ITensor(Op(op[i], v), s), ψ[v]) for (i, v) in enumerate(vertex_set)])
    )
  denominator_network = approx_network_region(ψψ, mts, [(v, 1) for v in vertex_set])

  if isnothing(sequence)
    sequence = contraction_sequence(numerator_network)
  end

  return contract(numerator_network; sequence)[] / contract(denominator_network; sequence)[]
end

function expect_BP(
  op::String,
  ψ::AbstractITensorNetwork,
  ψψ::AbstractITensorNetwork,
  mts::DataGraph;
  vertices=Graphs.vertices(ψ),
)
  ElT = promote_itensor_eltype(ψ)
  res = Dictionary(vertices, Vector{ElT}(undef, length(vertices)))
  for v in vertices
    res[v] = expect_BP(String[op], ψ, ψψ, mts, [v])
  end

  return res
end

function expect_BP(op::String, ψ::AbstractITensorNetwork; vertices=Graphs.vertices(ψ))
  ψψ = norm_network(ψ)
  mts = belief_propagation(
    ψψ,
    message_tensors(ψψ; subgraph_vertices=collect(values(group(v -> v[1], Graphs.vertices(ψψ)))));
    contract_kwargs=(; alg="exact"),
  )
  return expect_BP(op, ψ, ψψ, mts; vertices)
end

function expect_BP(
  op::String,
  ψ::AbstractITensorNetwork,
  ψψ::AbstractITensorNetwork;
  vertices=Graphs.vertices(ψ),
)
  mts = belief_propagation(
    ψψ,
    message_tensors(ψψ; subgraph_vertices=collect(values(group(v -> v[1], Graphs.vertices(ψψ)))));
    contract_kwargs=(; alg="exact"),
  )
  return expect_BP(op, ψ, ψψ, mts; vertices)
end