using ITensorNetworks: random_tensornetwork
using ITensorNetworks.ModelHamiltonians: ising, heisenberg
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using ITensors: expect
using Random

include("bp_dmrg.jl")
include("utils.jl")

Random.seed!(58484)


ITensors.disable_warn_order()

function main()
  g = heavy_hex_lattice_graph(2,2; periodic = true)
  #g = named_grid((12, 1); periodic = true)
  L = length(vertices(g))
  h, hlongitudinal, J = 0.6, 0.2, 1.0
  Δ = 0.5
  s = siteinds("S=1/2", g; conserve_qns = true)
  χ = 10
  #ψ0 = random_tensornetwork(s; link_space = 1)
  ψ0 = ITensorNetwork(v -> isodd(first(v)) ? "↑" : "↓", s)
  @show sum(expect(ψ0, "Z"))

  #H = ising(s; h, hl=hlongitudinal, J1=J)
  H = heisenberg(s; Δ)
  @show exact_heavy_hex_energy(ψ0, H) / L

  inserter_kwargs = (; maxdim = χ, cutoff = 1e-14)

  ψfinal, energies = bp_dmrg(ψ0, H; no_sweeps=5, nsites = 2, inserter_kwargs, forced_descent = false)

  final_energy_exact = exact_heavy_hex_energy(ψfinal, H) / L
  @show final_energy_exact

  @show sum(expect(ψfinal, "Z"))
  return final_mags = expect(ψfinal, "Z", ; alg="bp")
end

main()
