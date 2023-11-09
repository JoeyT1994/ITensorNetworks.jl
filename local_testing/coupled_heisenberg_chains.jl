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
  setindex_preserve_graph!,
  simple_update_bp,
  sqrt_and_inv_sqrt,
  simple_update_bp_full,
  expect_BP
using Dictionaries
using Observers
using NPZ
using Statistics

using SplitApplyCombine

using NamedGraphs: decorate_graph_edges

using OMEinsumContractionOrders

function heisenberg_chains(g::NamedGraph; J1=1.0, J2=0.0)
  ℋ = OpSum()
  if !iszero(J1)
    for e in edges(g)
      J = src(e)[1] != dst(e)[1] ? J2 : J1
      ℋ += J / 2, "S+", src(e), "S-", dst(e)
      ℋ += J / 2, "S-", src(e), "S+", dst(e)
      ℋ += J, "Sz", src(e), "Sz", dst(e)
    end
  end
  return ℋ
end

function calculate_Q2(
  ψ::AbstractITensorNetwork, ψψ::AbstractITensorNetwork, mts::DataGraph; central_row=true
)
  ny, nx = maximum(vertices(ψ))
  ny_iter = central_row ? [ceil(Int64, ny * 0.5)] : [i for i in 1:ny]
  nx_iter = [i for i in 1:nx]

  strings = [["X", "X"], ["Y", "Y"], ["Z", "Z"]]
  out = 0
  vertex_sets = [
    [(j, mod(i, nx) + 1), (j, mod(i + 1, nx) + 1)] for i in nx_iter for j in ny_iter
  ]

  for v_set in vertex_sets
    for string in strings
      out += expect_BP(string, ψ, ψψ, mts, v_set)
    end
  end

  return out / (length(nx_iter) * length(ny_iter))
end

function calculate_Q2_exact(
  ψ::ITensor, nx::Int64, ny::Int64, s::IndsNetwork; z=nothing, central_row=true
)
  ψ = copy(ψ)
  if isnothing(z)
    z = (ψ * dag(ψ))[]
  end

  ny_iter = central_row ? [ceil(Int64, ny * 0.5)] : [i for i in 1:ny]
  nx_iter = [i for i in 1:nx]

  strings = [["X", "X"], ["Y", "Y"], ["Z", "Z"]]
  out = 0
  vertex_sets = [
    [(j, mod(i, nx) + 1), (j, mod(i + 1, nx) + 1)] for i in nx_iter for j in ny_iter
  ]

  for v_set in vertex_sets
    for op_strings in strings
      oψ = copy(ψ)
      for (i, op_string) in enumerate(op_strings)
        oψ = noprime(oψ * op(op_string, s[v_set[i]]))
      end
      out += (oψ * dag(ψ))[]
    end
  end

  return out / (length(nx_iter) * length(ny_iter))
end

function calculate_Q3(
  ψ::AbstractITensorNetwork, ψψ::AbstractITensorNetwork, mts::DataGraph, central_row=true
)
  ny, nx = maximum(vertices(ψ))
  ny_iter = central_row ? [ceil(Int64, ny * 0.5)] : [i for i in 1:ny]
  nx_iter = [i for i in 1:nx]

  pos_strings = [["X", "Y", "Z"], ["Y", "Z", "X"], ["Z", "X", "Y"]]
  neg_strings = [["X", "Z", "Y"], ["Y", "X", "Z"], ["Z", "Y", "X"]]
  out = 0
  vertex_sets = [
    [(j, mod(i, nx) + 1), (j, mod(i + 1, nx) + 1), (j, mod(i + 2, nx) + 1)] for i in
                                                                                nx_iter for
    j in ny_iter
  ]

  for v_set in vertex_sets
    for pos_string in pos_strings
      out += expect_BP(pos_string, ψ, ψψ, mts, v_set)
    end
    for neg_string in neg_strings
      out -= expect_BP(neg_string, ψ, ψψ, mts, v_set)
    end
  end

  return out / (length(nx_iter) * length(ny_iter))
end

function calculate_Q3_exact(
  ψ::ITensor, nx::Int64, ny::Int64, s::IndsNetwork; z=nothing, central_row=true
)
  ψ = copy(ψ)
  if isnothing(z)
    z = (ψ * dag(ψ))[]
  end
  ny_iter = central_row ? [ceil(Int64, ny * 0.5)] : [i for i in 1:ny]
  nx_iter = [i for i in 1:nx]

  pos_strings = [["X", "Y", "Z"], ["Y", "Z", "X"], ["Z", "X", "Y"]]
  neg_strings = [["X", "Z", "Y"], ["Y", "X", "Z"], ["Z", "Y", "X"]]
  out = 0
  vertex_sets = [
    [(j, mod(i, nx) + 1), (j, mod(i + 1, nx) + 1), (j, mod(i + 2, nx) + 1)] for i in
                                                                                nx_iter for
    j in ny_iter
  ]

  for v_set in vertex_sets
    for pos_string in pos_strings
      oψ = copy(ψ)
      for (i, op_string) in enumerate(pos_string)
        oψ = noprime(oψ * op(op_string, s[v_set[i]]))
      end
      out += (oψ * dag(ψ))[]
    end
    for neg_string in neg_strings
      oψ = copy(ψ)
      for (i, op_string) in enumerate(neg_string)
        oψ = noprime(oψ * op(op_string, s[v_set[i]]))
      end
      out -= (oψ * dag(ψ))[]
    end
  end

  return out / (length(nx_iter) * length(ny_iter))
end

function exact_state_vector(ψ::ITensor, gate::ITensor; z=nothing)
  ψ = copy(ψ)
  if isnothing(z)
    z = (ψ * dag(ψ))[]
  end

  oψ = noprime!(ψ * gate)
  return (oψ * dag(ψ))[] / z
end

function grid_periodic_x(ny::Int64, nx::Int64)
  g = named_grid((ny, nx))
  for i in 1:ny
    NamedGraphs.add_edge!(g, (i, 1) => (i, nx))
  end
  return g
end

function get_boundaries(
  ψ::AbstractITensorNetwork; contract_norm_network=false, svd_kwargs...
)
  ψ = copy(ψ)
  ψψ = !contract_norm_network ? norm_network(ψ) : inner_network(ψ, ψ; combine_linkinds=true)
  vs = vertices(ψ)
  dims = maximum(vs)
  d1, d2 = dims
  half_d1 = floor(Int64, d1 / 2)
  row_iterators = [filter(v -> first(v) == i, vs) for i in 1:d1]
  alg = "ttn_svd"

  row_vs = if !contract_norm_network
    vcat([((v), 1) for v in row_iterators[d1]], [((v), 2) for v in row_iterators[d1]])
  else
    [v for v in row_iterators[d1]]
  end
  ψψR, _ = contract(
    ITensorNetwork(ITensor[ψψ[v] for v in row_vs]);
    alg=alg,
    output_structure=path_graph_structure,
    contraction_sequence_alg="sa_bipartite",
    cutoff=1e-8,
  )
  for i in reverse((half_d1 + 2):(d1 - 1))
    row_vs = if !contract_norm_network
      vcat([((v), 1) for v in row_iterators[i]], [((v), 2) for v in row_iterators[i]])
    else
      [v for v in row_iterators[i]]
    end
    ψψR_r, _ = contract(
      ITensorNetwork(ITensor[ψψ[v] for v in row_vs]);
      alg=alg,
      output_structure=path_graph_structure,
      contraction_sequence_alg="sa_bipartite",
      cutoff=1e-8,
    )
    ψψR = ψψR ⊗ ψψR_r
    ψψR, _ = contract(
      ψψR;
      alg=alg,
      output_structure=path_graph_structure,
      contraction_sequence_alg="sa_bipartite",
      svd_kwargs...,
    )
  end

  row_vs = if !contract_norm_network
    vcat([((v), 1) for v in row_iterators[1]], [((v), 2) for v in row_iterators[1]])
  else
    [v for v in row_iterators[1]]
  end
  ψψL, _ = contract(
    ITensorNetwork(ITensor[ψψ[v] for v in row_vs]);
    alg=alg,
    output_structure=path_graph_structure,
    contraction_sequence_alg="sa_bipartite",
    cutoff=1e-8,
  )
  for i in 2:half_d1
    row_vs = if !contract_norm_network
      vcat([((v), 1) for v in row_iterators[i]], [((v), 2) for v in row_iterators[i]])
    else
      [v for v in row_iterators[i]]
    end
    ψψL_r, _ = contract(
      ITensorNetwork(ITensor[ψψ[v] for v in row_vs]);
      alg=alg,
      output_structure=path_graph_structure,
      contraction_sequence_alg="sa_bipartite",
      cutoff=1e-8,
    )
    ψψL = ψψL ⊗ ψψL_r
    ψψL, _ = contract(
      ψψL;
      alg=alg,
      output_structure=path_graph_structure,
      contraction_sequence_alg="sa_bipartite",
      svd_kwargs...,
    )
  end

  middle_vs = if !contract_norm_network
    vcat(
      [((v), 1) for v in row_iterators[half_d1 + 1]],
      [((v), 2) for v in row_iterators[half_d1 + 1]],
    )
  else
    [v for v in row_iterators[half_d1 + 1]]
  end
  ψψM = ITensorNetwork(Dictionary(Dict(zip(middle_vs, ITensor[ψψ[v] for v in middle_vs]))))

  return ψψL, ψψR, ψψM
end

function main(
  g::NamedGraph, χparr::Int64, χperp::Int64, time_steps::Vector{Float64}, Jperp::Float64=0.0
)
  #s = siteinds("S=1/2", g; conserve_qns=true)
  s = siteinds("S=1/2", g)
  state_vector_backend = length(vertices(g)) < 30 ? true : false
  #state_vector_backend = true
  ny, nx = maximum(vertices(g))
  ITensors.disable_warn_order()

  ψ = ITensorNetwork(s, v -> if (v[2] + 2) % 3 == 0
    "X+"
  elseif (v[2] + 2) % 3 == 1
    "Y+"
  else
    "Z+"
  end)
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])
  mts = message_tensors(
    ψψ;
    subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ)))),
    itensor_constructor=denseblocks ∘ delta,
  )
  mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"))
  gates = heisenberg_chains(g; J2=Jperp)
  time = 0
  Q1s = zeros((length(time_steps) + 1))
  Q2s = zeros((length(time_steps) + 1))
  Q3s = zeros((length(time_steps) + 1))
  Q3s[1] = real(calculate_Q3(ψ, ψψ, mts))
  Q2s[1] = real(calculate_Q2(ψ, ψψ, mts))
  Q1s[1] = mean(collect(values(real.(expect_BP("Z", ψ, ψψ, mts)))))
  Q1s_exact, Q2s_exact, Q3s_exact = copy(Q1s), copy(Q2s), copy(Q3s)
  times = vcat([0.0], cumsum(time_steps))

  if state_vector_backend
    ψ_sv = reduce(*, ITensor[ψ[v] for v in vertices(ψ)])
  end

  for (i, dt) in enumerate(time_steps)
    @show time
    𝒰 = exp(-im * dt * gates; alg=Trotter{2}())
    u⃗ = Vector{ITensor}(𝒰, s)
    for u in u⃗
      v⃗ = ITensorNetworks._gate_vertices(u, ψ)
      e = NamedEdge(v⃗[1] => v⃗[2])
      χ = src(e)[1] != dst(e)[1] ? χperp : χparr
      ψ, ψψ, mts = apply(u, ψ, ψψ, mts; maxdim=χ, cutoff=1e-14)
      if state_vector_backend
        ψ_sv = noprime(ψ_sv * u)
      end
    end
    time += dt

    mts = belief_propagation(
      ψψ, mts; contract_kwargs=(; alg="exact"), niters=20, target_precision=1e-3
    )
    Q1s[i + 1] = mean(collect(values(real.(expect_BP("Z", ψ, ψψ, mts)))))
    Q2s[i + 1] = real(calculate_Q2(ψ, ψψ, mts))
    Q3s[i + 1] = real(calculate_Q3(ψ, ψψ, mts))
    if state_vector_backend
      Q2s_exact[i + 1] = real(calculate_Q2_exact(ψ_sv, nx, ny, s))
      Q3s_exact[i + 1] = real(calculate_Q3_exact(ψ_sv, nx, ny, s))
    end

    flush(stdout)
  end

  mts = belief_propagation(
    ψψ, mts; contract_kwargs=(; alg="exact"), niters=30, target_precision=1e-8
  )

  @show collect(values(real.(expect_BP("Z", ψ, ψψ, mts))))

  ΔQ1 = abs(Q1s[length(time_steps) + 1] - Q1s[1])
  ΔQ2 = abs(Q2s[length(time_steps) + 1] - Q2s[1])
  ΔQ3 = abs(Q3s[length(time_steps) + 1] - Q3s[1])

  println("Evolution finished. Change in conserved quantitys is $ΔQ1, $ΔQ2 and $ΔQ3.")

  return Q1s, Q2s, Q3s, Q1s_exact, Q2s_exact, Q3s_exact, times
end

if length(ARGS) > 1
  nx = parse(Int64, ARGS[1])
  ny = parse(Int64, ARGS[2])
  χparr = parse(Int64, ARGS[3])
  χperp = parse(Int64, ARGS[4])
  Jperp = parse(Float64, ARGS[5])
  save = true
else
  nx, ny = 36, 2
  Jperp = 0.05
  χparr, χperp = 64, 16
end

g = grid_periodic_x(ny, nx)

time_steps = [0.1 for i in 1:100]

@show χparr, χperp, Jperp
flush(stdout)
Q1s, Q2s, Q3s, Q1s_exact, Q2s_exact, Q3s_exact, times = main(
  g, χparr, χperp, time_steps, Jperp
)

save = true
if save
  file_str =
    "/mnt/home/jtindall/Documents/Data/ITensorNetworks/CoupledHeisenberg/ChiParr" *
    string(χparr) *
    "ChiPerp" *
    string(χperp) *
    "Nx" *
    string(nx) *
    "Ny" *
    string(ny) *
    "JPerp" *
    string(round(Jperp; digits=3)) *
    "Tmax" *
    string(round(sum(time_steps); digits=3))
  file_str *= ".npz"
  npzwrite(
    file_str;
    Q1s=Q1s,
    Q2s=Q2s,
    Q3s=Q3s,
    Q1s_exact=Q1s_exact,
    Q2s_exact=Q2s_exact,
    Q3s_exact=Q3s_exact,
    times=times,
  )
end
