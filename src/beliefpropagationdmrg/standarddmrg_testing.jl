using TOML
using ITensorNetworks: random_tensornetwork
using ITensorNetworks.ModelHamiltonians: ising, heisenberg
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using ITensors: expect
using Random
using DataGraphs: edge_data, vertex_data
using Dictionaries: Dictionary
using Graphs: nv, vertices
using ITensorMPS: ITensorMPS, MPS, randomMPS, dmrg, AbstractObserver
using ITensors: ITensors, ITensor, MPO, expect, Sweeps, setmaxdim!, setcutoff!
using KrylovKit: eigsolve
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using Observers: Observers, observer
using NamedGraphs.NamedGraphGenerators: named_grid
using NPZ

Base.@kwdef mutable struct ErrorObserver <: AbstractObserver
    total_sweeps::Int64
    no_bonds::Int64
    total_region_updates = 0
    region_energies = zeros(no_sweeps * 2 * no_bonds)
end

function ITensors.measure!(o::ErrorObserver; kwargs...)
    o.total_region_updates += 1
    o.region_energies[o.total_region_updates] = kwargs[:energy]
end

include("bp_dmrg.jl")
include("utils.jl")

save = false
graph_params = Dictionary(["L"], [16])
g, file_string = graph_parser("HYPERHONEYCOMB"; params = graph_params)

adj_mat = graph_to_adj_mat(g)
chi = 10

N = length(vertices(g))
s = siteinds("S=1/2", N)
Jx, Jy, Jz = 1.0, 1.0, 1.0
hx, hy, hz = 0.0, 0.0, 0.0
os = xyz_adjmat(N, adj_mat; Jx, Jy, Jz, hx, hy, hz)
H = MPO(os, s)
init_state = [isodd(i) ? "Up" : "Dn" for i = 1:N]
psi0 = randomMPS(s, init_state; linkdims =2)
no_sweeps = 10
sweeps = Sweeps(no_sweeps)
setmaxdim!(sweeps, chi)
setcutoff!(sweeps, 1E-14)

err_obs = ErrorObserver(total_sweeps = no_sweeps, no_bonds = N - 1)

e_f, psifinal =  dmrg(H,psi0, sweeps; observer = err_obs)
energies = (err_obs.region_energies) ./ N
final_mags = expect(psifinal, "Z")
@show e_f / N
@show sum(final_mags) / L

file_name =
"/Users/jtindall/Files/Data/DMRG/"*file_string*"Jx$(Jx)Jy$(Jy)Jz$(Jz)hx$(hx)hy$(hy)hz$(hz)chi$(chi)NoSweeps$(no_sweeps)"
if save
    npzwrite(file_name * ".npz"; energies=energies, final_mags=final_mags)
end

