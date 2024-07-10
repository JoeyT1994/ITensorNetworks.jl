using ITensorNetworks: random_tensornetwork
using ITensorNetworks.ModelHamiltonians: ising, heisenberg, xyz
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using ITensors: expect, disable_warn_order
using Random
using TOML

include("bp_dmrg.jl")
include("utils.jl")
include("treetensornetworkoperators.jl")

Random.seed!(58484)


disable_warn_order()

function main()

  file = "/Users/jtindall/Files/Data/BPDMRG/TOMLS/hyperhoneycomb.32.pbc.HB.Kitaev.nosyminfo.toml"
  data = TOML.parsefile(file)

  interactions = data["Interactions"]
  heisenberg_interactions = filter(d -> first(d) == "HB", interactions)

  g = build_graph_from_interactions(heisenberg_interactions)
  #g = lieb_lattice_graph(3,3; periodic = true)
  g = named_grid((5,5); periodic = true)
  g = named_hexagonal_lattice_graph(4,4; periodic = true)
  #g = heavy_hex_lattice_graph(2,2; periodic = true)
  #L = 6
  #g = named_grid((L, 1); periodic = true)
  L = length(vertices(g))
  Jx, Jy, Jz = 1.4, 0.0, -1.1
  hx, hy, hz = 0.5, 0.0, 0.2
  χ = 2
  s = siteinds("S=1/2", g; conserve_qns = false)
  ψ0 = random_tensornetwork(Float64, s; link_space =    χ)
  #ψ0 = ITensorNetwork(v -> isodd(first(v)) ? "↑" : "↓", s)

  H = xyz(s; Jx, Jy, Jz, hx, hy, hz)
  H = filter_zero_terms(H)

  inserter_kwargs = (; maxdim = χ, cutoff = 1e-14)

  ψfinal, energies = bp_dmrg(ψ0, H; no_sweeps=5, nsites = 1, inserter_kwargs)


end

main()
