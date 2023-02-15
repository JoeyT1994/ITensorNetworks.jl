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

function spin_gates(s::IndsNetwork, params; Δβ=0.1, rev=true)
  gates = ITensor[]
  for e in edges(s)
    n_srcv = length(neighbors(s, src(e)))
    n_dstv = length(neighbors(s, dst(e)))
    hj =
      4 * params["Jx"] * op("Sx", s[maybe_only(src(e))]) * op("Sx", s[maybe_only(dst(e))]) +
      4 *  params["Jy"] * op("Sy", s[maybe_only(src(e))]) * op("Sy", s[maybe_only(dst(e))]) +
      4 *  params["Jz"] * op("Sz", s[maybe_only(src(e))]) * op("Sz", s[maybe_only(dst(e))]) +
      2 * (params["hx"]/n_srcv) * op("Sx", s[maybe_only(src(e))]) * op("Id", s[maybe_only(dst(e))]) +
      2 * (params["hy"]/n_srcv) * op("Sy", s[maybe_only(src(e))]) * op("Id", s[maybe_only(dst(e))]) +
      2 * (params["hz"]/n_srcv) * op("Sz", s[maybe_only(src(e))]) * op("Id", s[maybe_only(dst(e))]) +
      2 * (params["hx"]/n_dstv) * op("Sx", s[maybe_only(dst(e))]) * op("Id", s[maybe_only(src(e))]) +
      2 * (params["hy"]/n_dstv) * op("Sy", s[maybe_only(dst(e))]) * op("Id", s[maybe_only(src(e))]) +
      2 * (params["hz"]/n_dstv) * op("Sz", s[maybe_only(dst(e))]) * op("Id", s[maybe_only(src(e))])
    if(Δβ != nothing)
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

function apply_gates(ψ::ITensorNetwork, gates, s::IndsNetwork; nvertices_per_partition = 1, maxdim, useenvs=  true)

  ψ = copy(ψ)
  for gate in gates
    v⃗ = neighbor_vertices(ψ, gate)
    ψ = ITensorNetworks.orthogonalize(ψ, v⃗[1])
    if(useenvs)
      ψψ = ψ ⊗ prime(dag(ψ); sites=[])
      vertex_groups = nested_graph_leaf_vertices(
        partition(partition(ψψ, group(v -> v[1], vertices(ψψ))); nvertices_per_partition)
      )
      mts = compute_message_tensors(ψψ; vertex_groups=vertex_groups)

      envs = get_environment(ψψ, mts, [(v⃗[1], 1), (v⃗[1], 2), (v⃗[2], 1), (v⃗[2], 2)])
    else
      envs = ITensor[]
    end
    ψ = apply(gate,ψ;maxdim, normalize=true,envs,envisposdef=true)

  end

  return ψ
end

function calc_energy(ψ::ITensorNetwork, params, s, maxdim)
  E = 0
  ψ = copy(ψ)
  Z = contract_inner(ψ, ψ)
  gates = spin_gates(s, params; Δβ = nothing, rev = false)

  for gate in gates
    Oψ = apply(gate, ψ; maxdim = 4*maxdim)
    E += contract_inner(ψ, Oψ)
  end

  return E/Z
end


    

function main()
  n = 5
  dims = (n, n)
  g = named_grid(dims)
  maxdim =2

  s = siteinds("S=1/2", g)
  params = Dict([("Jx", -0.8), ("Jy", 1.2), ("Jz", 0.0), ("hx", 0.0), ("hy", 0.0), ("hz", 0.0)])

  DMRG_backend(params, g, 4)
  ψ = ITensorNetwork(s, v -> "↑")
  ψSU = copy(ψ)
  ψBP = copy(ψ)

  betas = vcat([0.1 for i =1:10])
  for i = 1:length(betas)
    gates = spin_gates(s, params; Δβ=betas[i])
    println("On Sweep "*string(i))
    ψSU = apply_gates(ψSU, gates, s; maxdim, useenvs = false)
    ψBP = apply_gates(ψBP, gates, s; maxdim)
    ESU = calc_energy(ψSU, params, s, maxdim)
    EBP = calc_energy(ψBP, params, s, maxdim)
    @show ESU
    @show EBP
  end
end

ITensors.disable_warn_order()
main()