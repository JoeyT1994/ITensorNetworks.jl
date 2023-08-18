using NamedGraphs
using ITensorNetworks
using ITensors
using Random
using LinearAlgebra
using ITensorNetworks: contract_inner, neighbor_vertices, message_tensors, belief_propagation, symmetric_to_vidal_gauge, approx_network_region, full_update_bp, get_environment

using SplitApplyCombine

include("/mnt/home/jtindall/Documents/QuantumPhysics/JuliaCode/ExactDMRGBackend.jl")

function exp_ITensor(A::ITensor, beta::Union{Float64, ComplexF64}; nterms=10)
  #This should be identity when combinedind(inds(A, plev = 0)) = combinedind(inds(A, plev = 1))
  out = permute(A, [inds(A; plev=1)..., inds(A; plev=0)...])
  out = exp(0.0 * out)
  out = abs.(out)
  #Need to be VERY careful. The projector to up up has a minus sign in it!
  s1, s2, s3, s4 = inds(A; plev=1)[1],
  inds(A; plev=1)[2], inds(A; plev=0)[1],
  inds(A; plev=0)[2]
  out[s1 => 2, s2 => 2, s3 => 2, s4 => 2] = -1.0
  power = copy(out)
  for i in 1:nterms
    power = (1 / i) * swapprime(beta * power * prime(A), 2, 1)

    out = out + power
  end

  return out
end

function ITensor_to_matrix(A::ITensor)
  C_row, C_col = combiner(inds(A; plev=0)), combiner(inds(A; plev=1))
  A_mat = matrix(A * C_row * C_col, combinedind(C_row), combinedind(C_col))

  return A_mat
end

function Hubbard_gates(s::IndsNetwork; reverse_gates=true, imaginary_time=true, real_time = false, dbeta=-0.2)
  gates = ITensor[]
  for e in edges(s)
    vsrc, vdst = src(e), dst(e)
    hj = -op("Cdag", s[vsrc]) * op("C", s[vdst]) + op("C", s[vsrc]) * op("Cdag", s[vdst])
    if imaginary_time
      push!(gates, exp_ITensor(hj, dbeta / 2))
      #push!(gates, exp(-dbeta * hj /2))
    elseif real_time
      push!(gates, exp_ITensor(hj, 1.0*im*dbeta / 2))
    else
      push!(gates, hj)
    end
  end

  if reverse_gates
    append!(gates, reverse(gates))
  end

  return gates
end

function calc_energy(s::IndsNetwork, ψ::ITensorNetwork; seq)
  H_gates = Hubbard_gates(s; reverse_gates=false, imaginary_time=false)
  E = 0

  if isnothing(seq)
    ψψ = inner_network(ψ, ψ; combine_linkinds=true)
    seq = contraction_sequence(ψψ; alg="optimal")
  else
    seq = copy(seq)
  end

  for gate in H_gates
    ψO = evolve_su(ψ::ITensorNetwork, gate::ITensor; cutoff = 1e-16)
    E += contract_inner(ψO, ψ; sequence = seq)
  end

  return E / contract_inner(ψ, ψ; sequence = seq), seq
end

function evolve_su(ψ::ITensorNetwork, gate::ITensor; svd_kwargs...)

  ψ = copy(ψ)
  v⃗ = neighbor_vertices(ψ,  gate)
  e_ind = only(commoninds(ψ[v⃗[1]], ψ[v⃗[2]]))

  theta = noprime(ψ[v⃗[1]]*ψ[v⃗[2]]*gate)
  U, S, V = svd(theta, uniqueinds(ψ[v⃗[1]], ψ[v⃗[2]]); svd_kwargs...)
  US_ind = only(commoninds(U, S))
  ψ[v⃗[1]] = U
  ψ[v⃗[2]] = S*V
  new_ind = settags(US_ind, tags(e_ind))
  replaceind!(ψ[v⃗[1]], US_ind, new_ind)
  replaceind!(ψ[v⃗[2]], US_ind, new_ind)


  return ψ
end

function evolve_fu(ψ::ITensorNetwork, gate::ITensor; svd_kwargs...)

  ψ = copy(ψ)
  v⃗ = neighbor_vertices(ψ,  gate)

  mts, ψψ =  fermion_bp(ψ)

  @assert length(v⃗) == 2
  v1, v2 = v⃗

  envs = get_environment(ψψ, mts, [(v1, 1), (v1, 2), (v2, 1), (v2, 2)])

  envs = ITensor(envs)

  ψ[v1], ψ[v2] = full_update_bp(gate, ψ, v⃗; envs, svd_kwargs...)

  return ψ
end

function fermion_bp(ψ::ITensorNetwork)
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])

  mts = message_tensors(
    ψψ; subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ)))), itensor_constructor=inds_e -> delta(inds_e)
  )

  for e in edges(mts)
    mt_inds = [inds(mts[e][v]) for v in vertices(mts[e])]
    mts[e] = ITensorNetwork(delta(mt_inds))
  end

  mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"))

  return mts, ψψ
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
  ITensors.enable_auto_fermion()

  n = 12

  g = named_grid((n, 1))
  g = NamedGraphs.hexagonal_lattice_graph(2,2)
  s = siteinds("Fermion", g; conserve_qns=true)
  #add_edge!(g, (1, 1) => (n, 1))
  χ =2

  no_sweeps =10
  dbetas = [-0.1 for i in 1:no_sweeps]
  ψ = ITensorNetwork(s, v -> iseven(v[1] + v[2]) ? "Occ" : "Emp")
  #seq = nothing
  for i in 1:no_sweeps
    gates = Hubbard_gates(s; dbeta=dbetas[i], imaginary_time = false, real_time = true)
    for gate in gates
      #ψ = ITensorNetworks.apply(gate, ψ; maxdim=χ)
      ψ = evolve_su(ψ, gate; maxdim = χ, cutofff = 1e-16)
      #mts, ψψ =  fermion_bp(ψ)
      #ψ = evolve_fu(ψ, gate; maxdim = χ, cutofff = 1e-6)
    end
    #E, seq = calc_energy(s, ψ; seq)
    #@show E
  end

  mts, ψψ =  fermion_bp(ψ)

  @show [expect_state_SBP(op("N", s[v]), ψ, ψψ, mts) for v in vertices(g)]


  params = Dict([("U", 0.0), ("t", -1.0)])

  fermion_DMRG_backend(params, g, 50)

  return ITensors.disable_auto_fermion()
end

main()
