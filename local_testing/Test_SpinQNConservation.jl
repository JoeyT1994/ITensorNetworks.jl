using NamedGraphs
using ITensorNetworks
using ITensors
using Random
using LinearAlgebra
using ITensorNetworks: contract_inner, neighbor_vertices, message_tensors, belief_propagation, symmetric_to_vidal_gauge, approx_network_region, full_update_bp, get_environment, sgnpres_sqrt_diag, inv_sgnpres_sqrt_diag, sqrt_and_inv_sqrt, map_diag, inv_diag, find_subgraph, symmetric_factorize, sqrt_diag, diagblocks,
  initialize_bond_tensors, setindex_preserve_graph!, simple_update_bp
using Dictionaries
using Observers
using NPZ
using Statistics

using SplitApplyCombine
using Plots

using NamedGraphs: decorate_graph_edges

include("/mnt/home/jtindall/.julia/dev/ITensorNetworks/local_testing/DMRGBackend.jl")


function XXZ_gates(s::IndsNetwork; reverse_gates=true, imaginary_time=true, real_time = false, dbeta=-0.2, Δ = 1.0)
  gates = ITensor[]
  for e in edges(s)
    vsrc, vdst = src(e), dst(e)
    hj = -2.0*op("S+", s[vsrc]) * op("S-", s[vdst]) - 2.0*op("S-", s[vsrc]) * op("S+", s[vdst]) + Δ * op("Z", s[vsrc]) * op("Z", s[vdst])
    if imaginary_time
      push!(gates, exp(hj * dbeta / 2))
    elseif real_time
      push!(gates, exp(hj * 1.0*im*dbeta / 2))
    else
      push!(gates, hj)
    end
  end

  if reverse_gates
    append!(gates, reverse(gates))
  end

  return gates
end

function calc_energy(s::IndsNetwork, ψ::ITensorNetwork; seq, Δ = 1.0)
  gates = XXZ_gates(s; reverse_gates=false, imaginary_time=false, Δ)
  E = 0

  if isnothing(seq)
    ψψ = inner_network(ψ, ψ; combine_linkinds=true)
    seq = contraction_sequence(ψψ; alg="optimal")
  else
    seq = copy(seq)
  end

  for gate in gates
    ψO = apply(gate, ψ; cutoff=1e-16)
    E += contract_inner(ψO, ψ; sequence = seq)
  end

  z = contract_inner(ψ, ψ; sequence = seq)
  return E / z, seq
end

function evolve_fu(ψ::ITensorNetwork, gate::ITensor, mts::DataGraph, ψψ::ITensorNetwork; svd_kwargs...)

  ψ = copy(ψ)
  v⃗ = neighbor_vertices(ψ,  gate)

  @assert length(v⃗) == 2
  v1, v2 = v⃗

  s1 = find_subgraph((v1, 1), mts)
  s2 = find_subgraph((v2, 1), mts)
  obs = Observer()

  envs = get_environment(ψψ, mts, [(v1, 1), (v1, 2), (v2, 1), (v2, 2)])
  envs = Vector{ITensor}(envs)

  ψ[v⃗[1]], ψ[v⃗[2]] = simple_update_bp(gate, ψ, v⃗; envs, (observer!)=obs, svd_kwargs...)
  S = only(obs.singular_values)
  S /= norm(S)

  ψψ = norm_network(ψ)
  mts[s1] = ITensorNetwork{vertextype(mts[s1])}(dictionary([(v1, 1) => ψψ[v1, 1], (v1, 2) => ψψ[v1, 2]]))
  mts[s2] = ITensorNetwork{vertextype(mts[s2])}(dictionary([(v2, 1) => ψψ[v2, 1], (v2, 2) => ψψ[v2, 2]]))
  mts[s1 => s2] = ITensorNetwork(dag(S))
  mts[s2 => s1] = ITensorNetwork(S)



  return ψ, ψψ, mts
end

"""Take the expectation value of a an ITensor on an ITN using SBP"""
function expect_state_SBP(o::ITensor, ψ::AbstractITensorNetwork, ψψ::AbstractITensorNetwork, mts::DataGraph)
    Oψ = apply(o, ψ; cutoff = 1e-16)
    ψ = copy(ψ)
    s = siteinds(ψ)
    vs = vertices(s)[findall(i -> (length(commoninds(s[i], inds(o))) != 0), vertices(s))]
    vs_braket = [(v,1) for v in vs]

    numerator_network = approx_network_region(ψψ, mts, vs_braket; verts_tn=ITensorNetwork(ITensor[Oψ[v] for v in vs]))
    denominator_network = approx_network_region(ψψ, mts, vs_braket)
    num_seq = contraction_sequence(numerator_network; alg = "optimal")
    den_seq = contraction_sequence(numerator_network; alg = "optimal")
    return ITensorNetworks.contract(numerator_network; sequence = num_seq)[] / ITensorNetworks.contract(denominator_network, sequence = den_seq)[]

end


function main()

  n = 4
  g = named_grid((n, n))
  #add_edge!(g, (1,1) => (n,1))
  #g = NamedGraphs.hexagonal_lattice_graph(3,3)
  #g = decorate_graph_edges(g)

  g_vs = vertices(g)
  s = siteinds("S=1/2", g; conserve_qns=true)
  χ =4
  Δ = 0.2

  no_sweeps =5
  dbetas = [-0.05 for i in 1:no_sweeps]
  t_final = -sum(dbetas)
  ψ = ITensorNetwork(s, v -> findfirst(==(v), g_vs) % 2 == 0 ? "Dn" : "Up")
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])
  mts = message_tensors(
    ψψ; subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ)))), itensor_constructor= denseblocks ∘ delta)
  mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"))
  init_mags = [expect_state_SBP(op("Z", s[v]), ψ, ψψ, mts) for v in vertices(g)]

  seq = nothing
  for i in 1:no_sweeps
    #println("On Sweep $i")
    gates = XXZ_gates(s; dbeta=dbetas[i], imaginary_time = false, real_time = true, Δ)
    for gate in gates
      ψ, ψψ, mts = evolve_fu(ψ, gate, mts, ψψ; maxdim = χ, cutofff = 1e-16)
    end

    #mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"), niters = 20, target_precision = 1e-3)

    #ψ, ψψ, mts = re_gauge(ψ, ψψ, mts, s, χ; niters = 5)
    #E, seq = calc_energy(s, ψ; seq, Δ)
    #println("Current Energy $E")
  end

  println("Belief Propagation Evolution Finished")

  mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"), niters = 10, target_precision = 1e-5)

  final_mags =  real.([expect_state_SBP(op("Z", s[v]), ψ, ψψ, mts) for v in vertices(g)])

  @show init_mags
  @show final_mags

  #npzwrite("/mnt/home/jtindall/Documents/Data/ITensorNetworks/FreeFermions/"*lattice*"DynamicsBenchmarkBondDim"*string(χ)*"Tfinal"*string(round(t_final; digits =  2))*".npz", init_occs = init_occs, final_occs = final_occs, final_occs_exact = final_occs_exact)


  params = Dict([("J", -1.0), ("Δ", Δ)])
  #spin_DMRG_backend(params, g, 50)
  delta_t = 0.05
  spin_TDVP_backend(params, g, 100, t_final, delta_t)

end

main()
