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
  L = 20
  g = named_grid((L, 1); periodic = true)
  L = length(vertices(g))
  hamiltonian_params = (; hx = 0.5, hz = -0.4, Jx = 1.1, Jz = -0.3)
  χ = 2
  s = siteinds("S=1/2", g; conserve_qns = false)
  ψ0 = random_tensornetwork(s; link_space = χ)

  H = generic_spin_hamiltonian(s; hamiltonian_params...)

  ψfinal, energies = bp_dmrg(ψ0, H; no_sweeps=5, nsites = 1)


end

main()
