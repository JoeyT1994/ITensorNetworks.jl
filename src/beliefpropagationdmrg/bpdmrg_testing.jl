using TOML
using ITensorNetworks: random_tensornetwork
using ITensorNetworks.ModelHamiltonians: ising, heisenberg, xyz, xyzkitaev
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using NamedGraphs.GraphsExtensions: degrees
using ITensors: expect
using Random
using NPZ

include("bp_dmrg.jl")
include("utils.jl")

g = lieb_lattice_graph(3,3; periodic = true)
Jx, Jy, Jz = -0.5, 0.0, 1.3
hx, hy, hz = 1.4, 0.0, 0.2

Random.seed!(1234)

s = siteinds("S=1/2", g)
bp_update_kwargs= (; maxiter=25, tol=1e-9)

χ = 3
model_params = (; Jx, Jy, Jz, hx, hy, hz)
H = xyz(s; model_params...)

ψ0 = random_tensornetwork(s; link_space = χ)
no_sweeps = 5

inserter_kwargs = (; maxdim = χ, cutoff = 1e-16)
ψfinal, energies = bp_dmrg(ψ0, H; no_sweeps, nsites =1, bp_update_kwargs, inserter_kwargs)

