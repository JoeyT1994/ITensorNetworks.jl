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
  get_environment,
  find_subgraph,
  diagblocks,
  initialize_bond_tensors,
  setindex_preserve_graph!
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
  eigvals::Vector, U::Matrix, cidag_cj_init::Matrix, t::Float64
)
  Udag = adjoint(U)

  ckdag_cl_init = Udag * cidag_cj_init * U
  ckdag_cl_t =
    Diagonal(exp.(eigvals .* t .* im)) * ckdag_cl_init * Diagonal(exp.(-eigvals .* t .* im))

  cidag_cj_t = U * ckdag_cl_t * Udag
  return cidag_cj_t
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

  @show χ, lattice
  if lattice == "Chain"
    g = named_grid((4, 1); periodic=true)
    n = 250
    ginf = named_grid((n, 1))
    central_v = Tuple([ceil(Int, i / 2) for i in maximum(vertices(ginf))])
    central_v_index = findfirst(==(central_v), vertices(ginf))
    Ainf = Matrix(adjacency_matrix(ginf))
  elseif lattice == "2DSquare"
    g = NamedGraphs.named_grid((4, 4); periodic=true)
    n = 20
    ginf = NamedGraphs.named_grid((n, n))
    central_v = Tuple([ceil(Int, i / 2) for i in maximum(vertices(ginf))])
    central_v_index = findfirst(==(central_v), vertices(ginf))
    Ainf = Matrix(adjacency_matrix(ginf))
  elseif lattice == "2DHeavySquare"
    g = NamedGraphs.named_grid((2, 2))
    g = decorate_graph_edges(g)
    add_edge!(g, (1, 1) => (2, 1))
    add_edge!(g, (1, 1) => (1, 2))
    add_edge!(g, (2, 1) => (2, 2))
    add_edge!(g, (1, 2) => (2, 2))
    n = 20
    ginf = NamedGraphs.named_grid((n, n))
    central_v = Tuple([ceil(Int, i / 2) for i in maximum(vertices(ginf))])
    ginf = decorate_graph_edges(ginf)
    central_v_index = findfirst(==(central_v), vertices(ginf))
    Ainf = Matrix(adjacency_matrix(ginf))
  elseif lattice == "HeavyHexagon"
    g = NamedGraph([(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)])
    add_edges!(
      g,
      [
        (1, 1) => (1, 2),
        (1, 2) => (1, 3),
        (1, 2) => (2, 2),
        (2, 2) => (2, 3),
        (2, 3) => (1, 3),
        (2, 3) => (1, 1),
      ],
    )
    n = 20
    ginf = NamedGraphs.hexagonal_lattice_graph(n, n)
    ginf = decorate_graph_edges(ginf)
    init_occs = Dictionary(
      Dict(
        zip(
          vertices(g), [length(neighbors(g, v)) == 2 ? "Occ" : "Emp" for v in vertices(g)]
        ),
      ),
    )
    init_occs_inf = [length(neighbors(ginf, v)) == 2 ? 1.0 : 0.0 for v in vertices(ginf)]
    Ainf = Matrix(adjacency_matrix(ginf))
  elseif lattice == "Bethe"
    z = 2
    g = NamedGraph(Graphs.SimpleGraphs.complete_bipartite_graph(z, z))
    init_occs = Dictionary(
      Dict(zip(vertices(g), [v <= z ? "Occ" : "Emp" for v in vertices(g)]))
    )
    #nlayers = 11
    #ginf = bethe_lattice(z, nlayers)
    #init_occs_inf = [length(v) % 2 == 0 ? 1.0 : 0.0 for v in vertices(ginf)]
    #Ainf = Matrix(adjacency_matrix(ginf))
    n = 200
    ginf = named_grid((n, 1))
    init_occs_inf = [v[1] % 2 == 0 ? 1.0 : 0.0 for v in vertices(ginf)]
    Ainf =
      diagm(1 => [i == 1 ? sqrt(z) : sqrt(z - 1) for i in 1:(n - 1)]) + diagm(-1 => [i == 1 ? sqrt(z) : sqrt(z - 1) for i in 1:(n - 1)])
  elseif lattice == "3DCube"
    n = 10
    g = NamedGraphs.named_grid((4, 4, 4); periodic=true)
    ginf = NamedGraphs.named_grid((n, n, n))
    central_v = Tuple([ceil(Int, i / 2) for i in maximum(vertices(ginf))])
    central_v_index = findfirst(==(central_v), vertices(ginf))
    Ainf = Matrix(adjacency_matrix(ginf))
  end

  no_sweeps = 50
  dt = 0.1
  t_final = no_sweeps * dt
  dbetas = [-dt for i in 1:no_sweeps]
  g_vs = vertices(g)
  g_vs_inf = vertices(ginf)
  F = eigen(Ainf)
  eigvals, Umat = F.values, F.vectors
  s = siteinds("Fermion", g; conserve_qns=true)
  U = 0.0

  occs = zeros((no_sweeps + 1, length(g_vs)))
  occs_exact = zeros((no_sweeps + 1, length(g_vs_inf)))

  dbetas = [-dt for i in 1:no_sweeps]
  ψ = ITensorNetwork(s, v -> init_occs[v])

  ψψ = ψ ⊗ prime(dag(ψ); sites=[])
  mts = message_tensors(
    ψψ;
    subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ)))),
    itensor_constructor=denseblocks ∘ delta,
  )
  mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"))

  init_occs = [expect_state_SBP(op("N", s[v]), ψ, ψψ, mts) for v in vertices(g)]
  occs[1, :] = init_occs
  occs_exact[1, :] = init_occs_inf

  cidag_cj_init = Matrix(Diagonal(init_occs_inf))

  ITensors.disable_warn_order()

  seq = nothing
  for i in 1:no_sweeps
    println("On Sweep $i")
    gates = Hubbard_gates(
      s, U; dbeta=dbetas[i], imaginary_time=false, real_time=true, reverse_gates=true
    )
    for gate in gates
      #ψ, bond_tensors = apply_vidal_fermion(gate, ψ, bond_tensors)
      ψ, ψψ, mts = evolve_fu(ψ, gate, mts, ψψ; maxdim=χ, cutofff=1e-12)
      #mts, ψψ =  fermion_bp(ψ)
      #ψ = evolve_fu(ψ, gate; maxdim = χ, cutofff = 1e-6)
    end
    mts_temp = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"), niters=30)
    occs[i + 1, :] =
      real.([expect_state_SBP(op("N", s[v]), ψ, ψψ, mts_temp) for v in vertices(g)])
    cidag_cj = exact_dynamics_hopping_fermionic_model(eigvals, Umat, cidag_cj_init, i * dt)
    occs_exact[i + 1, :] = real.(diag(cidag_cj))

    #ψ, ψψ, mts = re_gauge(ψ, ψψ, mts, s, χ; niters = 5)
  end

  #ψ, _ = vidal_to_symmetric_gauge(ψ, bond_tensors)

  file_name = "/mnt/home/jtindall/Documents/Data/ITensorNetworks/FreeFermions/Inf" * lattice
  if lattice == "Bethe"
    file_name *= "z" * string(z)
  end
  npzwrite(
    file_name *
    "DynamicsBenchmarkBondDim" *
    string(χ) *
    "Tfinal" *
    string(round(t_final; digits=2)) *
    ".npz";
    occs=occs,
    occs_exact=occs_exact,
  )

  return ITensors.disable_auto_fermion()
end

χs = [[12, 24, 48, 96]]
lattices = ["Bethe"]
for (i, lattice) in enumerate(lattices)
  for (j, χ) in enumerate(flatten(χs[i, :]))
    main(χ, lattice)
  end
end
