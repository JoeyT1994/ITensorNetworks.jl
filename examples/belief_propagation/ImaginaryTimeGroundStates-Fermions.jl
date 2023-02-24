using ITensors
using ITensors: optimal_contraction_sequence
using ITensorVisualizationBase
using ITensorNetworks
using Statistics
using ITensorNetworks:
  orthogonalize,
  get_environment,
  rename_vertices_itn,
  nested_graph_leaf_vertices,
  compute_message_tensors,
  neighbor_vertices,
  contract_inner
using KaHyPar
using Dictionaries
using Compat
using Random
using LinearAlgebra
using NPZ
using NamedGraphs
using SplitApplyCombine
using CairoMakie

include("ExactDMRGBackend.jl")

maybe_only(x) = x
maybe_only(x::Tuple{T}) where {T} = only(x)

function dense_itn(ψ::ITensorNetwork)
  ψ = copy(ψ)
  for v in vertices(ψ)
    ψ[v] = dense(ψ[v])
  end
  return ψ
end

function fermion_gates(s::IndsNetwork, params; Δβ=0.1, rev=true)
  gates = ITensor[]
  for e in edges(s)
    hj =
      params["t"] * op("Cdag", s[maybe_only(src(e))]) * op("C", s[maybe_only(dst(e))]) +
      params["t"] * op("Cdag", s[maybe_only(dst(e))]) * op("C", s[maybe_only(src(e))])

    if (Δβ != nothing)
      Gj = exp(-Δβ * 0.5 * hj)
      push!(gates, Gj)
    else
      push!(gates, hj)
    end
  end

  if (rev)
    append!(gates, reverse(gates))
  end
  return gates
end

function apply_gates(
  ψ::ITensorNetwork, gates, s::IndsNetwork; nvertices_per_partition=1, maxdim, useenvs=true
)
  ψ = copy(ψ)
  for gate in gates
    v⃗ = neighbor_vertices(ψ, gate)
    ψ = ITensorNetworks.orthogonalize(ψ, v⃗[1])
    if (useenvs)
      ψψ = ψ ⊗ prime(dag(ψ); sites=[])
      vertex_groups = nested_graph_leaf_vertices(
        partition(partition(ψψ, group(v -> v[1], vertices(ψψ))); nvertices_per_partition)
      )
      mts = compute_message_tensors(ψψ; vertex_groups=vertex_groups)
      envs = get_environment(ψψ, mts, [(v⃗[1], 1), (v⃗[1], 2), (v⃗[2], 1), (v⃗[2], 2)])
    else
      envs = ITensor[]
    end

    ψ = apply(gate, ψ; maxdim, normalize=true, envs, envisposdef=true)
  end

  return ψ
end

function calc_energy(ψ::ITensorNetwork, params, s, maxdim)
  E = 0
  ψ = copy(ψ)
  Z = contract_inner(ψ, ψ)
  gates = fermion_gates(s, params; Δβ=nothing, rev=false)

  for gate in gates
    Oψ = apply(gate, ψ; maxdim=4 * maxdim)
    E += contract_inner(ψ, Oψ)
  end

  return E / Z
end

function main()
  n = 2
  dims = (n, 1)
  g = named_grid(dims)
  χ = 4
  ITensors.enable_auto_fermion()

  s = siteinds("Fermion", g; conserve_qns=true)
  params = Dict([("U", 0.0), ("t", -1.0)])

  DMRG_backend(params, g, χ, "Fermion")
  Random.seed!(2435)
  ψ = ITensorNetwork(s, v -> isodd(v[1]) ? "Emp" : "Occ")
  gates = fermion_gates(s, params; Δβ=0.1)
  nsweeps = 100

  for i in 1:nsweeps
    ψ = apply_gates(ψ, gates, s; maxdim=χ, useenvs=false)
    @show calc_energy(ψ, params, s, χ)
  end
end

main()
