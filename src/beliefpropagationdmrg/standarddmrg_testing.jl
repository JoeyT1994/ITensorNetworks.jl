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

include("utils.jl")

#g = lieb_lattice_graph(5,5; periodic = false)
g = named_grid((6,6))

adj_mat = graph_to_adj_mat(g)
chi = 250

N = length(vertices(g))
s = siteinds("S=1/2", N)
Jx, Jy, Jz = -0.5, 0.0, 1.3
hx, hy, hz = 1.4, 0.0, 0.2
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
#energies = (err_obs.region_energies) ./ N
@show e_f


