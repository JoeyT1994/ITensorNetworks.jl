using NamedGraphs
using ITensorNetworks
using ITensors
using Random
using LinearAlgebra
using ITensorNetworks: contract_inner, neighbor_vertices, message_tensors, belief_propagation, symmetric_to_vidal_gauge, approx_network_region, full_update_bp, get_environment, sgnpres_sqrt_diag, inv_sgnpres_sqrt_diag, sqrt_and_inv_sqrt, map_diag, inv_diag, find_subgraph, symmetric_factorize, sqrt_diag, diagblocks,
  initialize_bond_tensors, setindex_preserve_graph!, simple_update_bp, sqrt_and_inv_sqrt, simple_update_bp_full
using Dictionaries
using Observers
using NPZ
using Statistics

using SplitApplyCombine
using Plots

using NamedGraphs: decorate_graph_edges

"""Take the expectation value of a an ITensor on an ITN using SBP"""
function expect_state_SBP(o::ITensor, ψ::AbstractITensorNetwork, ψψ::AbstractITensorNetwork, mts::DataGraph)
    Oψ = apply(o, ψ; cutoff = 1e-16)
    ψ = copy(ψ)
    s = siteinds(ψ)
    vs = vertices(s)[findall(i -> (length(commoninds(s[i], inds(o))) != 0), vertices(s))]
    vs_braket = [(v,1) for v in vs]

    numerator_network = approx_network_region(ψψ, mts, vs_braket; verts_tn=ITensorNetwork(ITensor[Oψ[v] for v in vs]))
    denominator_network = approx_network_region(ψψ, mts, vs_braket)
    num_seq = contraction_sequence(numerator_network; alg = "optimal")
    den_seq = contraction_sequence(numerator_network; alg = "optimal")
    return ITensorNetworks.contract(numerator_network; sequence = num_seq)[] / ITensorNetworks.contract(denominator_network, sequence = den_seq)[]

end

function evolve_fu(ψ::ITensorNetwork, gate::ITensor, mts::DataGraph, ψψ::ITensorNetwork; apply_kwargs...)
  regularization = 1e-12

  ψ = copy(ψ)
  v⃗ = neighbor_vertices(ψ,  gate)
  e_ind = only(commoninds(ψ[v⃗[1]], ψ[v⃗[2]]))

  @assert length(v⃗) == 2
  v1, v2 = v⃗

  s1 = find_subgraph((v1, 1), mts)
  s2 = find_subgraph((v2, 1), mts)

  envs = get_environment(ψψ, mts, [(v1, 1), (v1, 2), (v2, 1), (v2, 2)])
  envs = Vector{ITensor}(envs)

  obs = Observer()
  ψᵥ₁, ψᵥ₂ = simple_update_bp(gate, ψ, v⃗; envs, (observer!)=obs, apply_kwargs...)

  S = only(obs.singular_values)

  ψᵥ₁ ./= norm(ψᵥ₁)
  ψᵥ₂ ./= norm(ψᵥ₂)

  ψ[v1], ψ[v2] = ψᵥ₁, ψᵥ₂

  ψψ = norm_network(ψ)
  mts[s1] = ITensorNetwork{vertextype(mts[s1])}(dictionary([(v1, 1) => ψψ[v1, 1], (v1, 2) => ψψ[v1, 2]]))
  mts[s2] = ITensorNetwork{vertextype(mts[s2])}(dictionary([(v2, 1) => ψψ[v2, 1], (v2, 2) => ψψ[v2, 2]]))
  mts[s1 => s2] = ITensorNetwork(dag(S))
  mts[s2 => s1] = ITensorNetwork(S)

  return ψ, ψψ, mts
end


function main(g::NamedGraph, χ::Int64, time_steps::Vector{Float64})

  g_vs = vertices(g)
  s = siteinds("S=1/2", g; conserve_qns=true)

  ψ = ITensorNetwork(s, v -> findfirst(==(v), g_vs) % 3 == 0 ? "Dn" : "Up")
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])
  mts = message_tensors(
    ψψ; subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ)))), itensor_constructor= denseblocks ∘ delta)
  mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"))
  init_mags = [expect_state_SBP(op("Z", s[v]), ψ, ψψ, mts) for v in vertices(g)]
  gates = heisenberg(g)
  time = 0

  println("Starting Sim. Bond dim is $χ")
  for dt in time_steps
    @show time
    𝒰 = exp(-dt * gates; alg=Trotter{2}())
    u⃗ = Vector{ITensor}(𝒰, s)
    for u in u⃗
      ψ, ψψ, mts = evolve_fu(ψ, u, mts, ψψ; maxdim = χ, cutofff = 1e-14)
    end
    time += dt

    mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"), niters = 20, target_precision = 1e-3)
  end

  mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"), niters = 30, target_precision = 1e-5)

  final_mags =  real.([expect_state_SBP(op("Z", s[v]), ψ, ψψ, mts) for v in vertices(g)])

  ΔM = abs(sum(init_mags) - sum(final_mags))
  println("Evolution finished. Change in conserved quantity is $ΔM.")

end

n = 20
g = named_grid((n, 1))
time_steps = [0.1 for i in 1:11]

main(g, 2, time_steps)
main(g, 4, time_steps)
main(g, 6, time_steps)