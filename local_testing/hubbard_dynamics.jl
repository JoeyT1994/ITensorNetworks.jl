using NamedGraphs
using Graphs
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

function doublon_op(s::Index)
  o = op("Nup", s)
  o = o * prime(op("Ndn", s))
  swapprime!(o, 2, 1)
  return o
end

function hubbard_gates(s::IndsNetwork; U::Float64=0.0, t::Float64=1.0, dt::Float64=0.1)
  gates = ITensor[]
  for e in edges(s)
    vsrc, vdst = src(e), dst(e)
    vsrc_z, vdst_z = length(neighbors(g, vsrc)), length(neighbors(g, vdst))
    if !iszero(t)
      hj =
        -t * op("Cdagup", s[vsrc]) * op("Cup", s[vdst]) +
        t * op("Cup", s[vsrc]) * op("Cdagup", s[vdst]) -
        t * op("Cdagdn", s[vsrc]) * op("Cdn", s[vdst]) +
        t * op("Cdn", s[vsrc]) * op("Cdagdn", s[vdst])
    end
    if !iszero(U)
      hj +=
        (U / vsrc_z) * op("Nupdn", s[vsrc]) * op("I", s[vdst]) +
        (U / vdst_z) * op("I", s[vsrc]) * op("Nupdn", s[vdst])
    end

    if !iszero(t) || !iszero(U)
      push!(gates, exp_ITensor(hj, dt * im / 2))
    end
  end
  append!(gates, reverse(gates))

  return gates
end

function main(
  g::NamedGraph, χ::Int64, Init_Occs, time_steps::Vector{Float64}, U::Float64=10.0
)
  s = siteinds("Electron", g; conserve_qns=true)
  ITensors.disable_warn_order()

  ψ = ITensorNetwork(s, v -> Init_Occs[v])
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])
  mts = message_tensors(
    ψψ;
    subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ)))),
    itensor_constructor=denseblocks ∘ delta,
  )
  mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"))
  time = 0
  doublon_occs = zeros((length(vertices(g)), length(time_steps) + 1))
  doublon_occs[:, 1] = collect(values(real.(expect_BP("Nupdn", ψ, ψψ, mts))))
  times = vcat([0.0], cumsum(time_steps))

  for (i, dt) in enumerate(time_steps)
    @show time
    u⃗ = hubbard_gates(s; U, dt)
    for (j, u) in enumerate(u⃗)
      ψ, ψψ, mts = apply(u, ψ, ψψ, mts; maxdim=χ, cutoff=1e-6)
    end
    time += dt

    mts = belief_propagation(
      ψψ, mts; contract_kwargs=(; alg="exact"), niters=250, target_precision=1e-3, verbose = true
    )
    doublon_occs[:, i + 1] = collect(values(real.(expect_BP("Nupdn", ψ, ψψ, mts))))
    flush(stdout)
  end

  return doublon_occs, times
end

ITensors.enable_auto_fermion()

if length(ARGS) > 1
  z = parse(Int64, ARGS[1])
  g = NamedGraph(Graphs.SimpleGraphs.complete_bipartite_graph(z, z))
  U = parse(Int64, ARGS[3])
  χ = parse(Int64, ARGS[2])
  save = true
else
  z = 3
  g = NamedGraph(Graphs.SimpleGraphs.complete_bipartite_graph(z, z))
  U = 5.0
  χ = 16
end

time_steps = [0.05 for i in 1:60]
init_occs = Dictionary(Dict(zip(vertices(g), [v <= z ? "Up" : "Dn" for v in vertices(g)])))

Us = [0.0]
χs = [256, 300, 400, 512]
for U in Us
  for χ in χs
    @show χ, z
    flush(stdout)
    local doublon_occs, times = main(g, χ, init_occs, time_steps, U)

    local save = true
    if save
      local file_str =
        "/mnt/home/jtindall/Documents/Data/ITensorNetworks/Hubbard/Chi" *
        string(χ) *
        "Z" *
        string(z) *
        "U" *
        string(round(U; digits=3)) *
        "Tmax" *
        string(round(sum(time_steps); digits=3))
      file_str *= ".npz"
      npzwrite(file_str; doublon_occs=doublon_occs, times=times)
    end
  end
end

ITensors.disable_auto_fermion()
