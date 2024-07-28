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

graph_params = Dictionary(["L"], [128])
#graph_params = Dictionary(["nx", "ny"], [4, 4])
g, file_string, kitaev_terms = graph_parser("HYPERHONEYCOMB"; params = graph_params)
L = length(vertices(g))
Jx, Jy, Jz = 1.0, 1.0, 1.0
hx, hy, hz = 0.0, 0.0, 0.0
K = 0.0
save = true
compute_exact_e = L <= 32 ? true : false
nsites =1

Random.seed!(1234)

s = siteinds("S=1/2", g)
bp_update_kwargs= (; maxiter=30, tol=1e-9, makeposdeffreq =1)

χ = 3
model_params = (; Jx, Jy, Jz, hx, hy, hz, K)
#H = xyz(s; model_params...)
H = xyzkitaev(s; kitaev_terms, model_params...)

apply_kwargs= (; cutoff=1e-14, maxdim=χ)
ψ0 = nsites == 2 ? random_tensornetwork(s; link_space = 1) : random_tensornetwork(s; link_space = χ)
#ψ0 = imaginary_time_evo(s,ψ0,xyz,dbetas;bp_update_kwargs,apply_kwargs, model_params)

exact_e = compute_exact_e ? exact_energy(ψ0, H) / L : 0
println("Initial Exact e is $exact_e")
no_sweeps = 3

inserter_kwargs = (; maxdim = χ, cutoff = 1e-14)
ψfinal, energies = bp_dmrg(ψ0, H; no_sweeps, nsites, bp_update_kwargs, inserter_kwargs)
final_mags = expect(ψfinal, "Z")
@show sum(final_mags / L)

exact_e = compute_exact_e ? exact_energy(ψfinal, H) / L : 0
println("Final Exact e is $exact_e")

file_name =
"/Users/jtindall/Files/Data/BPDMRG/"*file_string*"Jx$(Jx)Jy$(Jy)Jz$(Jz)hx$(hx)hy$(hy)hz$(hz)K$(K)chi$(χ)NoSweeps$(no_sweeps)"
if save
    npzwrite(file_name * ".npz"; energies=energies, final_mags=final_mags, exact_e = exact_e)
end

