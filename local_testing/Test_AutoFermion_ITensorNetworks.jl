using NamedGraphs
using ITensorNetworks
using ITensors
using Random
using LinearAlgebra
using ITensorNetworks:
  contract_inner,
  neighbor_vertices,
  message_tensors,
  belief_propagation,
  symmetric_to_vidal_gauge,
  approx_network_region,
  full_update_bp,
  get_environment,
  sqrt_and_inv_sqrt,
  find_subgraph,
  diagblocks,
  initialize_bond_tensors,
  setindex_preserve_graph!,
  group_commuting_itensors,
  simple_update_bp_full,
  simple_update_bp
using ITensors: map_diag!, map_diag
using Dictionaries
using Observers
using NPZ
using Statistics
using OMEinsumContractionOrders

using SplitApplyCombine
using Plots

using NamedGraphs: decorate_graph_edges, add_edges!

include("/mnt/home/jtindall/.julia/dev/ITensorNetworks/local_testing/DMRGBackend.jl")

function exp_ITensor(A::ITensor, beta::Union{Float64,ComplexF64}; nterms=10)
  #This should be identity when combinedind(inds(A, plev = 0)) = combinedind(inds(A, plev = 1))
  # out = permute(A, [inds(A; plev=1)..., inds(A; plev=0)...])
  # out = exp(0.0 * out)
  # out = abs.(out)
  # #Need to be VERY careful. The projector to up up has a minus sign in it!
  s1, s2, s3, s4 = inds(A; plev=1)[1],
  inds(A; plev=1)[2], inds(A; plev=0)[1],
  inds(A; plev=0)[2]
  # out[s1 => 2, s2 => 2, s3 => 2, s4 => 2] = -1.0
  out = dag(op("I", s3) * op("I", s4))
  power = copy(out)
  for i in 1:nterms
    power = (1 / i) * swapprime(beta * power * prime(A), 2, 1)

    out = out + power
  end

  return out
end

function Hubbard_gates(
  s::IndsNetwork, U; reverse_gates=true, imaginary_time=true, real_time=false, dbeta=-0.2
)
  gates = ITensor[]
  Random.seed!(124435)
  for e in edges(s)
    vsrc, vdst = src(e), dst(e)
    hj =
      -op("Cdag", s[vsrc]) * op("C", s[vdst]) +
      op("C", s[vsrc]) * op("Cdag", s[vdst]) +
      U * op("N", s[vsrc]) * op("N", s[vdst])
    if imaginary_time
      push!(gates, exp_ITensor(hj, dbeta / 2))
    elseif real_time
      if reverse_gates
        push!(gates, exp_ITensor(hj, 1.0 * im * dbeta / 2))
      else
        push!(gates, exp_ITensor(hj, 1.0 * im * dbeta))
      end
    else
      push!(gates, hj)
    end
  end

  if reverse_gates
    append!(gates, reverse(gates))
  end

  return gates
end

function calc_energy(s::IndsNetwork, ψ::ITensorNetwork; seq, U=0.0)
  H_gates = Hubbard_gates(s, U; reverse_gates=false, imaginary_time=false)
  E = 0

  if isnothing(seq)
    ψψ = inner_network(ψ, ψ; combine_linkinds=true)
    seq = contraction_sequence(ψψ; alg="optimal")
  else
    seq = copy(seq)
  end

  for gate in H_gates
    ψO = evolve_su(ψ::ITensorNetwork, gate::ITensor; cutoff=1e-16)
    E += contract_inner(ψO, ψ; sequence=seq)
  end

  z = contract_inner(ψ, ψ; sequence=seq)
  return E / z, seq
end

function evolve_su(ψ::ITensorNetwork, gate::ITensor; svd_kwargs...)
  ψ = copy(ψ)
  v⃗ = neighbor_vertices(ψ, gate)
  e_ind = only(commoninds(ψ[v⃗[1]], ψ[v⃗[2]]))

  theta = noprime(ψ[v⃗[1]] * ψ[v⃗[2]] * gate)
  ψ[v⃗[1]], ψ[v⃗[2]] = fermionic_asymmetric_factorize(
    theta, uniqueinds(ψ[v⃗[1]], ψ[v⃗[2]]); tags=tags(e_ind), svd_kwargs...
  )

  return ψ
end

function evolve_fu(
  ψ::ITensorNetwork, gate::ITensor, mts::DataGraph, ψψ::ITensorNetwork; apply_kwargs...
)
  ψ = copy(ψ)
  v⃗ = neighbor_vertices(ψ, gate)
  e_ind = only(commoninds(ψ[v⃗[1]], ψ[v⃗[2]]))

  @assert length(v⃗) == 2
  v1, v2 = v⃗

  s1 = find_subgraph((v1, 1), mts)
  s2 = find_subgraph((v2, 1), mts)

  envs = get_environment(ψψ, mts, [(v1, 1), (v1, 2), (v2, 1), (v2, 2)])
  envs = Vector{ITensor}(envs)

  obs = Observer()
  ψᵥ₁, ψᵥ₂ = simple_update_bp(gate, ψ, v⃗; envs, (observer!)=obs, apply_kwargs...)

  S = only(obs.singular_values)

  ψᵥ₁ ./= norm(ψᵥ₁)
  ψᵥ₂ ./= norm(ψᵥ₂)

  ψ[v1], ψ[v2] = ψᵥ₁, ψᵥ₂

  ψψ = norm_network(ψ)
  mts[s1] = ITensorNetwork{vertextype(mts[s1])}(
    dictionary([(v1, 1) => ψψ[v1, 1], (v1, 2) => ψψ[v1, 2]])
  )
  mts[s2] = ITensorNetwork{vertextype(mts[s2])}(
    dictionary([(v2, 1) => ψψ[v2, 1], (v2, 2) => ψψ[v2, 2]])
  )
  mts[s1 => s2] = ITensorNetwork(dag(S))
  mts[s2 => s1] = ITensorNetwork(S)

  return ψ, ψψ, mts
end

"""Take the expectation value of a an ITensor on an ITN using SBP"""
function expect_state_SBP(
  o::ITensor, ψ::AbstractITensorNetwork, ψψ::AbstractITensorNetwork, mts::DataGraph
)
  Oψ = apply(o, ψ; cutoff=1e-16)
  ψ = copy(ψ)
  s = siteinds(ψ)
  vs = vertices(s)[findall(i -> (length(commoninds(s[i], inds(o))) != 0), vertices(s))]
  vs_braket = [(v, 1) for v in vs]

  numerator_network = approx_network_region(
    ψψ, mts, vs_braket; verts_tn=ITensorNetwork(ITensor[Oψ[v] for v in vs])
  )
  denominator_network = approx_network_region(ψψ, mts, vs_braket)
  num_seq = contraction_sequence(numerator_network; alg="optimal")
  den_seq = contraction_sequence(numerator_network; alg="optimal")
  return ITensorNetworks.contract(numerator_network; sequence=num_seq)[] /
         ITensorNetworks.contract(denominator_network; sequence=den_seq)[]
end

function exact_dynamics_hopping_fermionic_model(
  A::Matrix, cidag_cj_init::Matrix, t::Float64
)
  F = eigen(A)
  eigvals, U = F.values, F.vectors
  Udag = adjoint(U)

  ckdag_cl_init = Udag * cidag_cj_init * U
  init_energy = sum(eigvals .* diag(ckdag_cl_init))
  ckdag_cl_t =
    Diagonal(exp.(eigvals .* t .* im)) * ckdag_cl_init * Diagonal(exp.(-eigvals .* t .* im))
  final_energy = sum(eigvals .* diag(ckdag_cl_t))

  cidag_cj_t = U * ckdag_cl_t * Udag
  return cidag_cj_t
end

function exact_state_vector(ψ::ITensor, gate::ITensor; z=nothing)
  ψ = copy(ψ)
  if isnothing(z)
    z = (ψ * dag(ψ))[]
  end

  oψ = noprime!(ψ * gate)
  return (oψ * dag(ψ))[] / z
end

function spawn_children!(
  g::NamedGraph, parent_node::Tuple, nlayers::Int64, no_children::Int64
)
  if length(parent_node) < nlayers
    children = [(parent_node..., i) for i in 1:no_children]
    for child in children
      add_vertex!(g, child)
      add_edge!(g, parent_node => child)
      spawn_children!(g, child, nlayers, no_children)
    end
  end
end

function bethe_lattice(z::Int64, nlayers::Int64)
  g = NamedGraph()
  parent = (1,)
  add_vertex!(g, parent)
  children = [(parent..., i) for i in 1:z]
  for child in children
    add_vertex!(g, child)
    add_edge!(g, parent => child)
    spawn_children!(g, child, nlayers, z - 1)
  end

  return g
end

function main(χ::Int64, lattice::String)
  ITensors.enable_auto_fermion()
  state_vector_backend = false
  save = true
  re_gauge = true

  @show χ, lattice

  if lattice == "Chain"
    n = 4
    g = named_grid((n, 1))
  elseif lattice == "ChainPBC"
    n = 12
    g = named_grid((n, 1))
    add_edge!(g, (1, 1) => (n, 1))
  elseif lattice == "Hexagonal"
    g = NamedGraphs.hexagonal_lattice_graph(4, 4)
  elseif lattice == "HeavyHexagonal"
    g = NamedGraphs.hexagonal_lattice_graph(4, 4)
    g = decorate_graph_edges(g)
  elseif lattice == "2DSquare"
    n = 4
    g = NamedGraphs.named_grid((n, n))
  elseif lattice == "3DCube"
    n = 3
    g = NamedGraphs.named_grid((n, n, n))
  elseif lattice == "CombTree"
    tooth_lengths = fill(6, 6)
    g = named_comb_tree(tooth_lengths)
  elseif lattice == "Bethe"
    z = 3
    nlayers = 5
    g = bethe_lattice(z, nlayers)
  end

  g_vs = vertices(g)

  A = Matrix(adjacency_matrix(g))
  s = siteinds("Fermion", g; conserve_qns=true)
  U = 0.0

  no_sweeps = 20
  dt = 0.05
  dbetas = [-dt for i in 1:no_sweeps]
  t_final = -sum(dbetas)
  ψ = ITensorNetwork(s, v -> findfirst(==(v), g_vs) % 2 == 0 ? "Occ" : "Emp")

  ψψ = ψ ⊗ prime(dag(ψ); sites=[])
  mts = message_tensors(
    ψψ;
    subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ)))),
    itensor_constructor=denseblocks ∘ delta,
  )
  mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"))

  init_occs = [expect_state_SBP(op("N", s[v]), ψ, ψψ, mts) for v in g_vs]
  @show init_occs
  cidag_cj_init = Matrix(Diagonal(init_occs))

  occs = zeros((no_sweeps + 1, length(g_vs)))
  occs_exact = zeros((no_sweeps + 1, length(g_vs)))
  occs[1, :], occs_exact[1, :] = init_occs, init_occs

  ITensors.disable_warn_order()

  if state_vector_backend
    ψ_sv = reduce(*, ITensor[ψ[v] for v in vertices(ψ)])
  end

  seq = nothing
  for i in 1:no_sweeps
    println("On Sweep $i")
    gates = Hubbard_gates(
      s, U; dbeta=dbetas[i], imaginary_time=false, real_time=true, reverse_gates=true
    )
    for gate in gates
      #ψ, bond_tensors = apply_vidal_fermion(gate, ψ, bond_tensors)
      ψ, ψψ, mts = evolve_fu(ψ, gate, mts, ψψ; maxdim=χ, cutofff=1e-12)
      if state_vector_backend
        ψ_sv = noprime(ψ_sv * gate)
      end
      #mts, ψψ =  fermion_bp(ψ)
      #ψ = evolve_fu(ψ, gate; maxdim = χ, cutofff = 1e-6)
    end

    if re_gauge
      mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"), niters=25)
      occs[i + 1, :] = real.([expect_state_SBP(op("N", s[v]), ψ, ψψ, mts) for v in g_vs])
    else
      mts_temp = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"), niters=25)
      occs[i + 1, :] =
        real.([expect_state_SBP(op("N", s[v]), ψ, ψψ, mts_temp) for v in g_vs])
    end
    cidag_cj = exact_dynamics_hopping_fermionic_model(A, cidag_cj_init, i * dt)
    occs_exact[i + 1, :] = real.(diag(cidag_cj))

    #ψ, ψψ, mts = re_gauge(ψ, ψψ, mts, s, χ; niters = 5)
    #E, seq = calc_energy(s, ψ; seq, U)
    #println("Current Energy $E")
  end

  #ψ, _ = vidal_to_symmetric_gauge(ψ, bond_tensors)
  #mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"), niters = 50)

  #occs[no_sweeps + 1, :] =  real.([expect_state_SBP(op("N", s[v]), ψ, ψψ, mts) for v in vertices(g)])

  @show mean(abs.(occs[no_sweeps + 1, :] - occs_exact[no_sweeps + 1, :]))

  if state_vector_backend
    final_occs_state_vec = [exact_state_vector(ψ_sv, op("N", s[v])) for v in g_vs]
    @show mean(abs.(occs_exact[no_sweeps + 1, :] - final_occs_state_vec))
  end

  if save
    file_str =
      "/mnt/home/jtindall/Documents/Data/ITensorNetworks/FreeFermions/" *
      lattice *
      "Nvertices" *
      string(length(g_vs)) *
      "DynamicsBenchmarkBondDim" *
      string(χ) *
      "Tfinal" *
      string(round(t_final; digits=2))
    if re_gauge
      file_str *= "Regauged"
    end
    file_str *= ".npz"
    npzwrite(file_str; occs=occs, occs_exact=occs_exact)
  end
  #ITensors.disable_auto_fermion()
  #params = Dict([("U", U), ("t", -1.0)])
  #delta_t = 0.025
  #fermion_TDVP_backend(params, g, 100, t_final, delta_t)

  return ITensors.disable_auto_fermion()
end

χs = [[24]]
lattices = ["CombTree"]
for (i, lattice) in enumerate(lattices)
  for (j, χ) in enumerate(flatten(χs[i, :]))
    main(χ, lattice)
  end
end
