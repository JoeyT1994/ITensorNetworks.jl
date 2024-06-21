using TOML
using ITensorNetworks: random_tensornetwork
using ITensorNetworks.ModelHamiltonians: ising, heisenberg, xyz
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using NamedGraphs.GraphsExtensions: degrees
using ITensors: expect
using Random
using NPZ

include("bp_dmrg.jl")
include("utils.jl")

file = "/Users/jtindall/Files/Data/BPDMRG/TOMLS/hyperhoneycomb.32.pbc.HB.Kitaev.nosyminfo.toml"
data = TOML.parsefile(file)

interactions = data["Interactions"]
heisenberg_interactions = filter(d -> first(d) == "HB", interactions)

g = build_graph_from_interactions(heisenberg_interactions)
L = 12
g = named_grid((L, 1); periodic = true)

χ = 3
s = siteinds("S=1/2", g)
Jx, Jy, Jz = 1.0, 1.0, 1.0
dbetas = [(50, 0.1), (50, 0.05), (50, 0.025), (50, 0.01)]
bp_update_kwargs= (; maxiter=20, tol=1e-8)
apply_kwargs= (; cutoff=1e-12, maxdim=χ)
#ψ0 = ITensorNetwork(v -> isodd(first(v)) ? "↑" : "↓", s)

ψ0 = random_tensornetwork(s; link_space = 1)

h, hl, J1 = 0.6, 0.2, 1.0
model_params = (; h, hl, J1)
ψf = imaginary_time_evo(s,ψ0,ising,dbetas;bp_update_kwargs,apply_kwargs, model_params)
