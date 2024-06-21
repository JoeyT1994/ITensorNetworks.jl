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
#g = renamer(named_grid((6, 1); periodic = true))
s = siteinds("S=1/2", g; conserve_qns = true)
K = 0
#H = heisenberg(s)
Jx, Jy, Jz = 1.0, 1.0, 1.0
L = length(vertices(g))
H = xyz(s; Jx, Jy, Jz)
save = false
χ = 3
use_optimizer = true

Random.seed!(1234)

s = siteinds("S=1/2", g)
K = 0
bp_update_kwargs= (; maxiter=20, tol=1e-8)

χ = 3
s = siteinds("S=1/2", g)
Jx, Jy, Jz = 1.0, 1.0, 1.0
dbetas = [(50, 0.1), (50, 0.05), (50, 0.025), (50, 0.01)]
bp_update_kwargs= (; maxiter=20, tol=1e-8)
apply_kwargs= (; cutoff=1e-12, maxdim=χ)
ψ0 = ITensorNetwork(v -> isodd(first(v)) ? "↑" : "↓", s)

#ψ0 = random_tensornetwork(s; link_space = 1)

model_params = (; Jx, Jy, Jz)
ψ0 = imaginary_time_evo(s,ψ0,xyz,dbetas;bp_update_kwargs,apply_kwargs, model_params)

#ψ0 = ITensorNetwork(v -> isodd(first(v)) ? "↑" : "↓", s)
no_sweeps = 3

inserter_kwargs = (; maxdim = χ, cutoff = 1e-14)
ψfinal, energies = bp_dmrg(ψ0, H; no_sweeps, nsites = 2, inserter_kwargs, optimizer = "Rotate")
final_mags = expect(ψfinal, "Z")

file_name =
"/Users/jtindall/Files/Data/BPDMRG/HYPERHONEYCOMB/XYZL$(L)Jx$(Jx)Jy$(Jy)Jz$(Jz)chi$(χ)NoSweeps$(no_sweeps)OPTIMISER"
file_name *= use_optimizer ? "ON" : "OFF"
if save
    npzwrite(file_name * ".npz"; energies=energies, final_mags=final_mags)
end

