using Compat
using ITensors
using Metis
using ITensorNetworks
using Random
using SplitApplyCombine
using Graphs
using NamedGraphs

using StatsBase
using Statistics

using ITensorNetworks:
  belief_propagation,
  approx_network_region,
  contract_inner,
  message_tensors,
  nested_graph_leaf_vertices,
  split_index,
  initialize_bond_tensors,
  vidal_itn_canonicalness,
  vidal_to_symmetric_gauge,
  vidal_gauge,
  norm_network

using NPZ
using Observers

ax_projs = Dict(
  zip(
    ["Z+", "X+", "Y+", "Z-", "X-", "Y-"],
    [[1.0, 0.0], [1.0, 1.0], [-1.0 * im, 1.0], [0.0, 1.0], [-1.0, 1.0], [1.0 * im, 1]],
  ),
)

function get_1site_rdm(ψ::ITensorNetwork, v)
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])
  Z = partition(ψψ; subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ)))))
  mts = belief_propagation(
    ψψ, message_tensors(Z); contract_kwargs=(; alg="exact"), target_precision=1e-8
  )
  ψψsplit = split_index(ψψ, NamedEdge.([(v, 1) => (v, 2)]))
  rdm = ITensors.contract(
    approx_network_region(
      ψψ, mts, [(v, 2)]; verts_tn=ITensorNetwork(ITensor[ψψsplit[(v, 2)]])
    ),
  )
  return rdm / tr(rdm)
end

function sample_itensornetwork_spinhalf(ψ::ITensorNetwork, s::IndsNetwork, v; ax="Z")
  rdm = get_1site_rdm(ψ, v)
  O = ITensors.op(ax, s[v])
  p = real(0.5 * ((O * rdm)[] + 1))
  if rand() < p
    return ax * "-"
  else
    return ax * "+"
  end
end

function one_site_proj_itensornetwork_spinhalf(ψ::ITensorNetwork, s::IndsNetwork, v; ax="Z")
  bit = sample_itensornetwork_spinhalf(ψ, s, v; ax)
  P = ITensor(ax_projs[bit], s[v])
  ψ[v] = (ψ[v] * dag(P)) * P
  return ψ, bit
end

function get_bitstring_spinhalf(ψ::ITensorNetwork, s::IndsNetwork)
  ψ = copy(ψ)
  axes = ["Z", "Y", "X"]
  bitstring = String[]
  for v in vertices(ψ)
    ψ, b = one_site_proj_itensornetwork_spinhalf(ψ, s, v; ax=rand(axes))
    push!(bitstring, b)
  end
  return bitstring
end

"""
Random field J1-J2 Heisenberg model on a general graph
"""
function heisenberg(
  g::NamedGraph;
  hx::Vector{Float64}=zeros((length(vertices(g)))),
  hy::Vector{Float64}=zeros((length(vertices(g)))),
  hz::Vector{Float64}=zeros((length(vertices(g)))),
)
  ℋ = OpSum()
  for e in edges(g)
    ℋ += -0.5, "S+", src(e), "S-", dst(e)
    ℋ += -0.5, "S-", src(e), "S+", dst(e)
    ℋ += -1.0, "Sz", src(e), "Sz", dst(e)
  end

  for (i, v) in enumerate(vertices(g))
    if !iszero(hx[i])
      ℋ -= hx[i], "Sx", v
    end
    if !iszero(hy[i])
      ℋ -= hy[i], "Sy", v
    end
    if !iszero(hz[i])
      ℋ -= hz[i], "Sz", v
    end
  end
  return ℋ
end

"""Take the expectation value of o on an ITN using belief propagation"""
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
  return ITensors.contract(numerator_network; sequence=num_seq)[] /
         ITensors.contract(denominator_network; sequence=den_seq)[]
end

function main(;
  χ::Int64=8,
  z::Int64=3,
  nverts::Int64=10,
  W::Float64=1.0,
  dβ::Float64=0.025,
  T::Float64=1.0,
  nMETTS::Int64=10,
  nwarmupMETTS::Int64=10,
  id::Int64=1,
  save=true,
)
  β = 1.0 / T
  g = NamedGraph(Graphs.random_regular_graph(nverts, z))
  s = siteinds("S=1/2", g)
  n_dbetas = ceil(Int64, β / dβ)
  target_c = 1e-3
  xmags = zeros((nMETTS + nwarmupMETTS, length(vertices(g))))
  ymags = zeros((nMETTS + nwarmupMETTS, length(vertices(g))))
  zmags = zeros((nMETTS + nwarmupMETTS, length(vertices(g))))
  errs = zeros((nMETTS + nwarmupMETTS))
  projs = ["Z+", "X+", "Y+", "Z-", "X-", "Y-"]
  Random.seed!(1234 * id)

  gates = heisenberg(
    g; hx=W .* randn((nverts)), hy=W .* randn((nverts)), hz=W .* randn((nverts))
  )
  𝒰 = exp(-dβ * gates; alg=Trotter{2}())
  ψ = ITensorNetwork(s, v -> rand(projs))
  u⃗ = Vector{ITensor}(𝒰, s)
  for k in 1:(nMETTS + nwarmupMETTS)
    e = 1.0
    flush(stdout)
    if k <= nwarmupMETTS
      println("On Warm Up METT $k")
    else
      println("On METT $(k - nwarmupMETTS)")
    end
    bond_tensors = initialize_bond_tensors(ψ)
    for i in 1:n_dbetas
      for u in u⃗
        obs = Observer()
        ψ, bond_tensors = apply(u, ψ, bond_tensors; (observer!) = obs, normalize=true, maxdim=χ, cutoff=1e-12)
        e *= (1.0 - obs.truncerr[])
      end

      cur_C = vidal_itn_canonicalness(ψ, bond_tensors)
      while cur_C >= target_c
        for e in edges(ψ)
          ψ, bond_tensors = apply(e, ψ, bond_tensors; normalize=true, cutoff=1e-16)
        end
        cur_C = vidal_itn_canonicalness(ψ, bond_tensors)
      end
    end
    e = e ^ (1.0/ (n_dbetas * length(u⃗)))
    @show maxlinkdim(ψ)

    ψ, mts = vidal_to_symmetric_gauge(ψ, bond_tensors)
    ψψ = norm_network(ψ)
    xmags[k, :] = Float64[
      real.(expect_state_SBP(op("X", s[v]), ψ, ψψ, mts)) for v in vertices(ψ)
    ]
    ymags[k, :] = Float64[
      real.(expect_state_SBP(op("Y", s[v]), ψ, ψψ, mts)) for v in vertices(ψ)
    ]
    zmags[k, :] = Float64[
      real.(expect_state_SBP(op("Z", s[v]), ψ, ψψ, mts)) for v in vertices(ψ)
    ]
    @show mean(
      xmags[k, :] + ymags[k, :] + zmags[k, :]
    )

    if k != nMETTS + nwarmupMETTS
      bits = get_bitstring_spinhalf(ψ, s)
      bits = Dict(zip([v for v in vertices(ψ)], bits))
      ψ = ITensorNetwork(s, v -> bits[v])
    end

    println("Avg. Gate fidelity from this METT was $e")
    errs[k] = e
  end

  file_str = "/mnt/home/jtindall/Documents/Data/ITensorNetworks/Heisenberg_RRG/"
  file_str *=
    "METTSz" *
    string(z) *
    "nv" *
    string(nverts) *
    "T" *
    string(round(T; digits=3)) *
    "W" *
    string(round(W; digits=3)) *
    "χ" *
    string(χ) *
    "nMETTS" *
    string(nMETTS) *
    "nwarmupMETTS" *
    string(nwarmupMETTS) *
    "id" *
    string(id)
  file_str *= ".npz"
  if save
    npzwrite(file_str; xmags=xmags, ymags=ymags, zmags=zmags, errs = errs)
  end
end

if length(ARGS) > 1
  χ = parse(Int64, ARGS[1])
  z = parse(Int64, ARGS[2])
  nverts = parse(Int64, ARGS[3])
  T = parse(Float64, ARGS[4])
  W = parse(Float64, ARGS[5])
  nMETTS = parse(Int64, ARGS[6])
  nwarmupMETTS = parse(Int64, ARGS[7])
  id = parse(Int64, ARGS[8])
  save = true
else
  χ, z, nverts, T, W, id, nMETTS, nwarmupMETTS = 16, 3, 150, 1.0, 0.0, 1, 0, 1
  save = false
end

main(; χ, z, nverts, T, W, id, nMETTS, nwarmupMETTS, save, dβ = 0.1)
