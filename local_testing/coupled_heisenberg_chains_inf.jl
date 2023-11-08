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

function heisenberg_chains(g::NamedGraph; J1=1.0, J2=0.0)
  ℋ = OpSum()
  for e in edges(g)
    J = src(e)[1] != dst(e)[1] ? J2 : J1
    if !iszero(J)
      ℋ += J / 2, "S+", src(e), "S-", dst(e)
      ℋ += J / 2, "S-", src(e), "S+", dst(e)
      ℋ += J, "Sz", src(e), "Sz", dst(e)
    end
  end
  return ℋ
end

function expect_BP_splitindex(
  op::Vector{String},
  ψ::AbstractITensorNetwork,
  ψψ::AbstractITensorNetwork,
  mts::DataGraph,
  vertex_set,
  e;
  sequence=nothing,
)
  s = siteinds(ψ)
  @assert length(vertex_set) == length(op)
  ψ = copy(ψ)
  ψψ = copy(ψψ)
  mts = copy(mts)
  vsrc, vdst = src(e), dst(e)
  e_ind = only(linkinds(ψ, e))
  e_ind_sim = sim(e_ind)
  ψ[vdst] = replaceind(ψ[vdst], e_ind, e_ind_sim)
  ψψ[(vdst, 1)] = replaceind(ψψ[(vdst, 1)], e_ind, e_ind_sim)
  ψψ[(vdst, 2)] = replaceind(ψψ[(vdst, 2)], e_ind', e_ind_sim')

  numerator_network = approx_network_region(
    ψψ,
    mts,
    [(v, 1) for v in vertex_set];
    verts_tn=ITensorNetwork(
      ITensor[apply(ITensor(Op(op[i], v), s), ψ[v]) for (i, v) in enumerate(vertex_set)]
    ),
  )

  denominator_network = approx_network_region(ψψ, mts, [(v, 1) for v in vertex_set])

  mt_1 = ITensor(mts[find_subgraph((vdst, 1), mts) => find_subgraph((vsrc, 1), mts)])
  mt_2 = ITensor(mts[find_subgraph((vsrc, 1), mts) => find_subgraph((vdst, 1), mts)])
  mt_2 = replaceind(mt_2, e_ind, e_ind_sim)
  mt_2 = replaceind(mt_2, e_ind', e_ind_sim')

  numerator_network = ITensorNetwork(vcat(ITensor(numerator_network), ITensor[mt_1, mt_2]))
  denominator_network = ITensorNetwork(
    vcat(ITensor(denominator_network), ITensor[copy(mt_1), copy(mt_2)])
  )

  if isnothing(sequence)
    sequence = contraction_sequence(numerator_network)
  end

  out = contract(numerator_network; sequence)[] / contract(denominator_network; sequence)[]
  return out
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
      out += expect_BP_splitindex(
        pos_string, ψ, ψψ, mts, v_set, NamedEdge(first(v_set) => last(v_set))
      )
    end
    for neg_string in neg_strings
      out -= expect_BP_splitindex(
        neg_string, ψ, ψψ, mts, v_set, NamedEdge(first(v_set) => last(v_set))
      )
    end
  end

  return out / (length(nx_iter) * length(ny_iter))
end

function main(
  g::NamedGraph, χparr::Int64, χperp::Int64, time_steps::Vector{Float64}, Jperp::Float64=0.0
)
  #s = siteinds("S=1/2", g; conserve_qns=true)
  s = siteinds("S=1/2", g)
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
  times = vcat([0.0], cumsum(time_steps))

  for (i, dt) in enumerate(time_steps)
    @show time
    𝒰 = exp(-im * dt * gates; alg=Trotter{2}())
    u⃗ = Vector{ITensor}(𝒰, s)
    for u in u⃗
      v⃗ = ITensorNetworks._gate_vertices(u, ψ)
      e = NamedEdge(v⃗[1] => v⃗[2])
      χ = src(e)[1] != dst(e)[1] ? χperp : χparr
      ψ, ψψ, mts = apply(u, ψ, ψψ, mts; maxdim=χ, cutoff=1e-12)
    end
    time += dt

    mts = belief_propagation(
      ψψ, mts; contract_kwargs=(; alg="exact"), niters=20, target_precision=1e-3
    )
    Q1s[i + 1] = mean(collect(values(real.(expect_BP("Z", ψ, ψψ, mts)))))
    Q2s[i + 1] = real(calculate_Q2(ψ, ψψ, mts))
    Q3s[i + 1] = real(calculate_Q3(ψ, ψψ, mts))

    flush(stdout)
  end

  mts = belief_propagation(
    ψψ, mts; contract_kwargs=(; alg="exact"), niters=30, target_precision=1e-5
  )

  @show collect(values(real.(expect_BP("Z", ψ, ψψ, mts))))

  ΔQ1 = abs(Q1s[length(time_steps) + 1] - Q1s[1])
  ΔQ2 = abs(Q2s[length(time_steps) + 1] - Q2s[1])
  ΔQ3 = abs(Q3s[length(time_steps) + 1] - Q3s[1])

  println("Evolution finished. Change in conserved quantitys is $ΔQ1, $ΔQ2 and $ΔQ3.")

  return Q1s, Q2s, Q3s, times
end

if length(ARGS) > 1
  χparr = parse(Int64, ARGS[1])
  χperp = parse(Int64, ARGS[2])
  Jperp = parse(Float64, ARGS[3])
  save = true
else
  Jperp = 0.0
  χparr, χperp = 64, 1
end

g = named_grid((3, 3); periodic=true)
time_steps = [0.1 for i in 1:200]

@show χparr, χperp
flush(stdout)
Q1s, Q2s, Q3s, times = main(g, χparr, χperp, time_steps, Jperp)

save = true
if save
  file_str =
    "/mnt/home/jtindall/Documents/Data/ITensorNetworks/CoupledHeisenberg/ChiParr" *
    string(χparr) *
    "ChiPerp" *
    string(χperp) *
    "NxInfNyInfJPerp" *
    string(round(Jperp; digits=3)) *
    "Tmax" *
    string(round(sum(time_steps); digits=3))
  file_str *= ".npz"
  npzwrite(file_str; Q1s=Q1s, Q2s=Q2s, Q3s=Q3s, times=times)
end
