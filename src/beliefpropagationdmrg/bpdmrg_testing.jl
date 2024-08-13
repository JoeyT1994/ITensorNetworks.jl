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

g = lieb_lattice_graph(6,6; periodic = false)
#g = named_hexagonal_lattice_graph(10,10)
Jx, Jy, Jz = -0.5, 0.0, 1.3
hx, hy, hz = 1.4, 0.0, 0.2

Random.seed!(1234)

s = siteinds("S=1/2", g)
bp_update_kwargs= (; maxiter=25, tol=1e-9)

ITensors.disable_warn_order()

χ = 2
model_params = (; Jx, Jy, Jz, hx, hy, hz)
H = xyz(s; model_params...)

dbetas = [(100, 0.1), (100, 0.05)]
bp_update_kwargs= (; maxiter=20, tol=1e-8)
apply_kwargs= (; cutoff=1e-14, maxdim=χ)
ψ0 = random_tensornetwork(s; link_space = 2)
ψIψ_bpc = BeliefPropagationCache(QuadraticFormNetwork(ψ0))

ψ0, ψIψ_bpc = renormalize_update_norm_cache(ψ0, ψIψ_bpc; cache_update_kwargs=bp_update_kwargs)

cur_local_state, ∂ψOψ_bpc_∂rs, ∂ψIψ_bpc_∂r = bp_extracter(ψ0, H, ψIψ_bpc, [last(vertices(ψ0))]; dist = 0)

E = sum([contract([∂ψOψ_bpc_∂r; cur_local_state; dag(prime(cur_local_state))])[] for ∂ψOψ_bpc_∂r in ∂ψOψ_bpc_∂rs]) / contract([∂ψIψ_bpc_∂r; cur_local_state; dag(prime(cur_local_state))])[]
e = sum(expect(ψ0, H; alg="bp"))

@show E, e
#ψf = imaginary_time_evo(s,ψ0,xyz,dbetas;bp_update_kwargs,apply_kwargs, model_params)

#inserter_kwargs = (; maxdim = χ, cutoff = 1e-16)
#dist=  6
#println("Moving to R = $dist")
#ψR0, energies = bp_dmrg(ψf, H; no_sweeps = 2, nsites =1, bp_update_kwargs, inserter_kwargs, dist)

