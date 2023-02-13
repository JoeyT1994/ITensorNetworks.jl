using ITensors
using ITensors: optimal_contraction_sequence
using ITensorVisualizationBase
using ITensorNetworks
using Statistics
using ITensorNetworks:
  orthogonalize,
  norm_sqr,
  construct_initial_mts,
  get_environment,
  rename_vertices_itn,
  calculate_contraction
using KaHyPar
using Dictionaries
using Compat
using Random
using LinearAlgebra
using NPZ
using NamedGraphs
using SplitApplyCombine
using CairoMakie

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
    Gj = exp(-Δβ * 0.5 * hj)
    push!(gates, Gj)
  end

  if (rev)
    append!(gates, reverse(gates))
  end
  return gates
end

function electron_gates(s::IndsNetwork, params; Δβ=0.1, rev=true)
  gates = ITensor[]
  for e in edges(s)
    hj =
      params["U"] * op("Nup", s[maybe_only(src(e))]) * op("Ndn", s[maybe_only(dst(e))]) -
      params["t"] * op("Cdagup", s[maybe_only(src(e))]) * op("Cup", s[maybe_only(dst(e))]) +
      params["t"] * op("Cdagdn", s[maybe_only(src(e))]) * op("Cdn", s[maybe_only(dst(e))])
    Gj = exp(-Δβ * 0.5 * hj)
    push!(gates, Gj)
  end

  if (rev)
    append!(gates, reverse(gates))
  end
  return gates
end


n = 4
dims = (n, n)
g = named_grid(dims)
s = siteinds("Electron", g)

#params = Dict([("Jx", -0.8), ("Jy", -0.0), ("Jz", 0.0), ("hx", 0.0), ("hy", 0.0), ("hz", 2.2)])

#gates = spin_gates(s, params)

params = Dict([("U", 5.0), ("t", -1.0)])
gates = electron_gates(s, params)

#@show length(gates)
#@show length(edges(g))