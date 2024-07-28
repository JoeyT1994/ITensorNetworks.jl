using DataGraphs: edge_data, vertex_data
using Dictionaries: Dictionary
using Graphs: nv, vertices
using ITensorMPS: ITensorMPS, randomMPS
using ITensors: ITensors, ITensor, MPO, expect, Sweeps, setmaxdim!, setcutoff!, dmrg
using KrylovKit: eigsolve
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using Observers: observer
using NamedGraphs.NamedGraphGenerators: named_grid

using NPZ

include("utils.jl")

Random.seed!(5634)

function graph_to_adj_mat(g::AbstractGraph)
  L = length(vertices(g))
  adj_mat = zeros((L, L))
  for (i, v1) in enumerate(vertices(g))
    for (j, v2) in enumerate(vertices(g))
      if NamedEdge(v1 => v2) ∈ edges(g) || NamedEdge(v2 => v1) ∈ edges(g)
        adj_mat[i, j] = 1
      end
    end
  end
  return adj_mat
end

function ising_adjmat(L, adj_mat; J, h, hl)
  os = OpSum()
  for i in 1:L
    for j in (i + 1):L
      if !iszero(J * adj_mat[i, j])
        os += J * adj_mat[i, j], "Sz", i, "Sz", j
      end
    end
  end
  for i in 1:L
    os += h, "Sx", i
    os += hl, "Sz", i
  end

  return os
end

function heisenberg_adjmat(L, adj_mat; J=1, Δ=1)
  os = OpSum()
  for i in 1:L
    for j in (i + 1):L
      if !iszero(J * adj_mat[i, j])
        os += J * adj_mat[i, j] / 2, "S+", i, "S-", j
        os += J * adj_mat[i, j] / 2, "S-", i, "S+", j
        os += J * adj_mat[i, j] * Δ, "Sz", i, "Sz", j
      end
    end
  end
  return os
end

function main()
  graph = "PBCChain"
  L = 12
  if graph == "PBCChain"
    g = named_grid((L, 1); periodic=true)
  elseif graph == "PBCHeavyHex"
    g = heavy_hex_lattice_graph(L, L; periodic=true)
  end
  adj_mat = graph_to_adj_mat(g)
  save = false
  chi = 25

  N = length(vertices(g))
  h, hl = 0.6, 0.2
  Δ = 0.5
  J = 1
  s = siteinds("S=1/2", N; conserve_qns=false)

  h, hlongitudinal, J = 0.6, 0.2, 1.0
  os = ising_adjmat(N, adj_mat; J, h, hl)
  H = MPO(os, s)

  #psi0 = ttn(random_tensornetwork(s; link_space=chi))
  init_state = [isodd(i) ? "Up" : "Dn" for i in 1:N]
  psi0 = randomMPS(s, init_state; linkdims=2)
  no_sweeps = 20
  sweeps = Sweeps(no_sweeps)
  setmaxdim!(sweeps, chi)
  setcutoff!(sweeps, 1E-14)

  e_f, psifinal = dmrg(H, psi0, sweeps)
  @show e_f / N

  file_name =
    "/Users/jtindall/Files/Data/DMRG/ISING" *
    graph *
    "L$(N)h$(h)hl$(hl)J$(J)chi$(chi)NoSweeps$(no_sweeps)"
  if save
    npzwrite(file_name * ".npz"; energies=energies, final_mags=final_mags)
  end
end

main()
