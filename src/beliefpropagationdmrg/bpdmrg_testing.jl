using TOML
using ITensorNetworks: random_tensornetwork
using ITensorNetworks.ModelHamiltonians: ising, heisenberg, xyz, xyzkitaev
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using NamedGraphs.GraphsExtensions: degrees, eccentricity
using ITensors: expect
using Random
using NPZ
using Graphs: center
using Statistics
using OMEinsumContractionOrders

include("bp_dmrg.jl")
include("utils.jl")


g = lieb_lattice_graph(6,6; periodic = true)
Jx, Jy, Jz = -0.5, 0.0, 1.3
hx, hy, hz = 1.4, 0.0, 0.2

Random.seed!(1294)

s = siteinds("S=1/2", g)
bp_update_kwargs= (; maxiter=25, tol=1e-9)

ITensors.disable_warn_order()

χ = 3
model_params = (; Jx, Jy, Jz, hx, hy, hz)
H = xyz(s; model_params...)

dbetas = [(100, 0.1), (100, 0.05)]
bp_update_kwargs= (; maxiter=30, tol=1e-12)
apply_kwargs= (; cutoff=1e-14, maxdim=χ)
ψ0 = random_tensornetwork(s; link_space = 2)
ψIψ_bpc = BeliefPropagationCache(QuadraticFormNetwork(ψ0))
ψ0, ψIψ_bpc = renormalize_update_norm_cache(ψ0, ψIψ_bpc; cache_update_kwargs=bp_update_kwargs)
Es = []
R = 0
for v in vertices(ψ0)
    local cur_local_state, ∂ψOψ_bpc_∂rs, ∂ψIψ_bpc_∂r = bp_extracter(ψ0, H, ψIψ_bpc, [v]; dist = R)
    local e = 0
    for ∂ψOψ_bpc_∂r in ∂ψOψ_bpc_∂rs
        local tensors = ITensor[cur_local_state; dag(prime(cur_local_state)); ∂ψOψ_bpc_∂r]
        local seq = contraction_sequence(tensors; alg = "sa_bipartite")
        local o = contract(tensors; sequence = seq)[]
        e += o
    end
    local tensors = ITensor[cur_local_state; dag(prime(cur_local_state)); ∂ψIψ_bpc_∂r]
    local seq = contraction_sequence(tensors; alg = "sa_bipartite")
    e /= contract(tensors; sequence = seq)[]
    push!(Es, e)
end

@show var(Es)
@show mean(Es)
#ψf = imaginary_time_evo(s,ψ0,xyz,dbetas;bp_update_kwargs,apply_kwargs, model_params)

#inserter_kwargs = (; maxdim = χ, cutoff = 1e-16)
#dist=  6
#println("Moving to R = $dist")
#ψR0, energies = bp_dmrg(ψf, H; no_sweeps = 2, nsites =1, bp_update_kwargs, inserter_kwargs, dist)

