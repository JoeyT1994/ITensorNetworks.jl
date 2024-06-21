using ITensorNetworks: random_tensornetwork
using ITensorNetworks.ModelHamiltonians: ising, heisenberg
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using ITensors: expect, disable_warn_order
using Random

include("bp_dmrg.jl")
include("utils.jl")
include("treetensornetworkoperators.jl")

Random.seed!(58484)


disable_warn_order()

function main()
  g = heavy_hex_lattice_graph(4,4; periodic = true)
  L = 12
  g = named_grid((L, 1); periodic = true)
  L = length(vertices(g))
  h, hlongitudinal, J = 0.6, 0.2, 1.0
  χ = 3
  s = siteinds("S=1/2", g; conserve_qns = false)
  ψ0 = random_tensornetwork(s; link_space = 1)

  H = ising(s; h, hl=hlongitudinal, J1=J)

  inserter_kwargs = (; maxdim = χ, cutoff = 1e-14)

  ψfinal, energies = bp_dmrg(ψ0, H; no_sweeps=5, nsites = 2, inserter_kwargs)


end

main()
