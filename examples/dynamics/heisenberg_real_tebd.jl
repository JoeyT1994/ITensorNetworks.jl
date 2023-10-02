using NamedGraphs
using ITensorNetworks
using ITensors
using Random
using LinearAlgebra
using ITensorNetworks:
  contract_inner,
  neighbor_vertices,
  message_tensors,
  belief_propagation,
  symmetric_to_vidal_gauge,
  approx_network_region,
  full_update_bp,
  get_environment,
  find_subgraph,
  diagblocks,
  initialize_bond_tensors,
  setindex_preserve_graph!,
  simple_update_bp,
  sqrt_and_inv_sqrt,
  simple_update_bp_full,
  expect_BP
using Dictionaries
using Observers
using NPZ
using Statistics

using SplitApplyCombine
using Plots

using NamedGraphs: decorate_graph_edges

function main(g::NamedGraph, χ::Int64, time_steps::Vector{Float64})
  g_vs = vertices(g)
  s = siteinds("S=1/2", g; conserve_qns=true)

  ψ = ITensorNetwork(s, v -> findfirst(==(v), g_vs) % 2 == 0 ? "Dn" : "Up")
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])
  mts = message_tensors(
    ψψ;
    subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ)))),
    itensor_constructor=denseblocks ∘ delta,
  )
  mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"))
  init_mags = real.(expect_BP("Z", ψ, ψψ, mts))
  gates = heisenberg(g)
  time = 0

  println("Starting Sim. Bond dim is $χ")
  for dt in time_steps
    @show time
    𝒰 = exp(-im * dt * gates; alg=Trotter{2}())
    u⃗ = Vector{ITensor}(𝒰, s)
    for u in u⃗
      ψ, ψψ, mts = apply(u, ψ, ψψ, mts; maxdim=χ, cutoff=1e-14)
    end
    time += dt

    mts = belief_propagation(
      ψψ, mts; contract_kwargs=(; alg="exact"), niters=20, target_precision=1e-3
    )
  end

  mts = belief_propagation(
    ψψ, mts; contract_kwargs=(; alg="exact"), niters=30, target_precision=1e-5
  )

  final_mags = real.(expect_BP("Z", ψ, ψψ, mts))

  ΔM = abs(sum(init_mags) - sum(final_mags))
  return println("Evolution finished. Change in conserved quantity is $ΔM.")
end

n = 3
g = NamedGraphs.hexagonal_lattice_graph(n, n)
time_steps = [0.1 for i in 1:11]

#main(g, 2, time_steps)
main(g, 4, time_steps)
