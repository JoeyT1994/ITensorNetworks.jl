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

L = 32
save = true
file = "/Users/jtindall/Files/Data/BPDMRG/TOMLS/hyperhoneycomb.$(L).pbc.HB.Kitaev.nosyminfo.toml"
data = TOML.parsefile(file)

interactions = data["Interactions"]
heisenberg_interactions = filter(d -> first(d) == "HB", interactions)

g = build_graph_from_interactions(heisenberg_interactions)
#g = renamer(named_grid((6, 1); periodic = true))
adj_mat = graph_to_adj_mat(g)
chi = 2

N = length(vertices(g))
K = 0
Jx, Jy, Jz = 1.8, 0.9, -1.2
if Jx == Jy == Jz
    s = siteinds("S=1/2", N; conserve_qns = true)
    os = heisenberg_adjmat(N, adj_mat)
else
    s = siteinds("S=1/2", N; conserve_qns = false)
    os = xyz_adjmat(N, adj_mat; Jx = 1.0, Jy = 1.0, Jz = 1.0)
end
H = MPO(os, s)
init_state = [isodd(i) ? "Up" : "Dn" for i = 1:N]
psi0 = randomMPS(s, init_state; linkdims =2)
no_sweeps = 10
sweeps = Sweeps(no_sweeps)
#setmaxdim!(sweeps,trunc(Int, chi/4), trunc(Int, chi/4), trunc(Int, chi/2), trunc(Int, chi/2),trunc(Int, chi), trunc(Int, chi))
setmaxdim!(sweeps, chi)
setcutoff!(sweeps, 1E-14)

err_obs = ErrorObserver(total_sweeps = no_sweeps, no_bonds = N - 1)

e_f, psifinal =  dmrg(H,psi0, sweeps; observer = err_obs)
energies = (err_obs.region_energies) ./ N
final_mags = expect(psifinal, "Z")
@show e_f / N
@show sum(final_mags) / L

file_name =
"/Users/jtindall/Files/Data/DMRG/HYPERHONEYCOMB/XYZL$(N)Jx$(Jx)Jy$(Jy)Jz$(Jz)chi$(chi)NoSweeps$(no_sweeps)"
if save
    npzwrite(file_name * ".npz"; energies=energies, final_mags=final_mags)
end

