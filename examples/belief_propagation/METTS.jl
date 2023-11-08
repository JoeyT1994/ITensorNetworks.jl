using Compat
using ITensors
using Metis
using ITensorNetworks
using Random
using SplitApplyCombine
using NamedGraphs

using StatsBase
using Statistics

using ITensorNetworks:
  belief_propagation,
  approx_network_region,
  contract_inner,
  message_tensors,
  nested_graph_leaf_vertices,
  split_index,
  initialize_bond_tensors,
  vidal_itn_canonicalness,
  vidal_to_symmetric_gauge,
  vidal_gauge,
  norm_network

ax_projs = Dict(
  zip(
    ["Z+", "X+", "Y+", "Z-", "X-", "Y-"],
    [[1.0, 0.0], [1.0, 1.0], [-1.0 * im, 1.0], [0.0, 1.0], [-1.0, 1.0], [1.0 * im, 1]],
  ),
)

function get_1site_rdm(ψ::ITensorNetwork, v)
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])
  Z = partition(ψψ; subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ)))))
  mts = belief_propagation(
    ψψ, message_tensors(Z); contract_kwargs=(; alg="exact"), target_precision=1e-8
  )
  ψψsplit = split_index(ψψ, NamedEdge.([(v, 1) => (v, 2)]))
  rdm = contract(
    approx_network_region(
      ψψ, mts, [(v, 2)]; verts_tn=ITensorNetwork(ITensor[ψψsplit[(v, 2)]])
    ),
  )
  return rdm / tr(rdm)
end

function sample_itensornetwork(ψ::ITensorNetwork, v)
  rdm = get_1site_rdm(ψ, v)
  return StatsBase.sample([i for i in 1:length(diag(rdm))], Weights(real.(diag(rdm))))
end

function sample_itensornetwork_spinhalf(ψ::ITensorNetwork, s::IndsNetwork, v; ax="Z")
  rdm = get_1site_rdm(ψ, v)
  O = ITensors.op(ax, s[v])
  p = real(0.5 * ((O * rdm)[] + 1))
  if rand() < p
    return ax * "-"
  else
    return ax * "+"
  end
end

function one_site_proj_itensornetwork(ψ::ITensorNetwork, s::IndsNetwork, v)
  bit = sample_itensornetwork(ψ, v)
  P = ITensor([i == bit ? 1 : 0 for i in 1:dim(s[v])], s[v])
  ψ[v] = noprime!((ψ[v] * P) * P')
  return ψ, bit
end

function one_site_proj_itensornetwork_spinhalf(ψ::ITensorNetwork, s::IndsNetwork, v; ax="Z")
  bit = sample_itensornetwork_spinhalf(ψ, s, v; ax)
  P = ITensor(ax_projs[bit], s[v])
  ψ[v] = noprime!((ψ[v] * P) * P')
  return ψ, bit
end

function get_bitstring(ψ::ITensorNetwork, s::IndsNetwork)
  ψ = copy(ψ)
  bitstring = Int64[]
  for v in vertices(ψ)
    ψ, b = one_site_proj_itensornetwork(ψ, s, v)
    push!(bitstring, b)
  end
  return bitstring
end

function get_bitstring_spinhalf(ψ::ITensorNetwork, s::IndsNetwork)
  ψ = copy(ψ)
  axes = ["Z", "Y", "X"]
  bitstring = String[]
  for v in vertices(ψ)
    ψ, b = one_site_proj_itensornetwork_spinhalf(ψ, s, v; ax=rand(axes))
    push!(bitstring, b)
  end
  return bitstring
end

"""Take the expectation value of o on an ITN using belief propagation"""
function expect_state_SBP(
  o::ITensor, ψ::AbstractITensorNetwork, ψψ::AbstractITensorNetwork, mts::DataGraph
)
  Oψ = apply(o, ψ; cutoff=1e-16)
  ψ = copy(ψ)
  s = siteinds(ψ)
  vs = vertices(s)[findall(i -> (length(commoninds(s[i], inds(o))) != 0), vertices(s))]
  vs_braket = [(v, 1) for v in vs]

  numerator_network = approx_network_region(
    ψψ, mts, vs_braket; verts_tn=ITensorNetwork(ITensor[Oψ[v] for v in vs])
  )
  denominator_network = approx_network_region(ψψ, mts, vs_braket)
  num_seq = contraction_sequence(numerator_network; alg="optimal")
  den_seq = contraction_sequence(numerator_network; alg="optimal")
  return contract(numerator_network; sequence=num_seq)[] /
         contract(denominator_network; sequence=den_seq)[]
end

function main()
  g_dims = (1, 8)
  g = named_grid(g_dims)
  s = siteinds("S=1/2", g)
  n_dbetas, dβ, nMETTS, nwarmupMETTS = 60, 0.02, 250, 10
  χ, target_c = 8, 1e-3
  mags = zeros((nMETTS, length(vertices(g))))
  projs = ["Z+", "X+", "Y+", "Z-", "X-", "Y-"]

  gates = ising(g; h=1.08)
  𝒰 = exp(-dβ * gates; alg=Trotter{2}())
  ψ = ITensorNetwork(s, v -> rand(projs))
  u⃗ = Vector{ITensor}(𝒰, s)
  for k in 1:(nMETTS + nwarmupMETTS)
    if k <= nwarmupMETTS
      println("On Warm Up METT $k")
    else
      println("On METT $(k - nwarmupMETTS)")
    end
    bond_tensors = initialize_bond_tensors(ψ)
    for i in 1:n_dbetas
      for u in u⃗
        ψ, bond_tensors = apply(u, ψ, bond_tensors; normalize=true, maxdim=χ)
      end

      cur_C = vidal_itn_canonicalness(ψ, bond_tensors)
      if cur_C >= target_c
        ψ, _ = vidal_to_symmetric_gauge(ψ, bond_tensors)
        ψ, bond_tensors = vidal_gauge(ψ; niters=20, cutoff=1e-14)
      end
    end

    if k > nwarmupMETTS
      ψ, mts = vidal_to_symmetric_gauge(ψ, bond_tensors)
      ψψ = norm_network(ψ)
      mags[k - nwarmupMETTS, :] = Float64[
        real.(expect_state_SBP(op("X", s[v]), ψ, ψψ, mts)) for v in vertices(ψ)
      ]
      @show mags[k - nwarmupMETTS, 2]
    end

    bits = get_bitstring_spinhalf(ψ, s)
    bits = Dict(zip([v for v in vertices(ψ)], bits))
    ψ = ITensorNetwork(s, v -> bits[v])
  end

  @show mean([mean(mags[:, i]) for i in 1:length(vertices(g))])
end

main()
