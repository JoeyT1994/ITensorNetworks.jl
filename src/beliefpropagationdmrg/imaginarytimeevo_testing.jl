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

s = siteinds("S=1/2", g; conserve_qns = true)
L = length(vertices(g))
Jx, Jy, Jz = -0.5, 0.0, 1.3
hx, hy, hz = 1.4, 0.0, 0.2

χ = 3
s = siteinds("S=1/2", g)
model_params = (; Jx, Jy, Jz, hx, hy, hz)
dbetas = [(100, 0.1), (100, 0.05), (100, 0.025), (100, 0.01)]
bp_update_kwargs= (; maxiter=25, tol=1e-8, makeposdeffreq = 5)
apply_kwargs= (; cutoff=1e-14, maxdim=χ)

ψ0 = random_tensornetwork(s; link_space = 1)
ψf = imaginary_time_evo(s,ψ0,xyzkitaev,dbetas;bp_update_kwargs,apply_kwargs, model_params)
