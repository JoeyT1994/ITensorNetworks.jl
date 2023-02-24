using ITensors
using NamedGraphs
using Graphs

function HSpin(adj_mat, params, sites)
  ampo = OpSum()
  @assert size(adj_mat)[1] == size(adj_mat)[2]
  L = size(adj_mat)[1]
  for i in 1:L
    for j in (i + 1):L
      ampo += 4 * adj_mat[i, j] * params["Jx"], "Sx", i, "Sx", j
      ampo += 4 * adj_mat[i, j] * params["Jy"], "Sy", i, "Sy", j
      ampo += 4 * adj_mat[i, j] * params["Jz"], "Sz", i, "Sz", j
    end
  end

  for i in 1:L
    ampo += 2 * params["hx"], "Sx", i
    ampo += 2 * params["hy"], "Sy", i
    ampo += 2 * params["hz"], "Sz", i
  end
  H = MPO(ampo, sites)
  return H
end

function Helectron(adj_mat, params, sites)
  ampo = OpSum()
  @assert size(adj_mat)[1] == size(adj_mat)[2]
  L = size(adj_mat)[1]
  for i in 1:L
    for j in (i + 1):L
      ampo += adj_mat[i, j] * params["t"], "Cdagup", i, "Cup", j
      ampo += adj_mat[i, j] * params["t"], "Cdagdn", i, "Cdn", j
      ampo += adj_mat[i, j] * params["t"], "Cdagup", j, "Cup", i
      ampo += adj_mat[i, j] * params["t"], "Cdagdn", j, "Cdn", i
    end
  end

  for i in 1:L
    ampo += params["U"], "Nup", i, "Ndn", i
  end

  H = MPO(ampo, sites)
  return H
end

function Hfermion(adj_mat, params, sites)
  ampo = OpSum()
  @assert size(adj_mat)[1] == size(adj_mat)[2]
  L = size(adj_mat)[1]
  for i in 1:L
    for j in (i + 1):L
      ampo += adj_mat[i, j] * params["t"], "Cdag", i, "C", j
      ampo += adj_mat[i, j] * params["t"], "Cdag", j, "C", i
    end
  end

  H = MPO(ampo, sites)
  return H
end

function DMRG_backend(params, g::AbstractGraph, χ::Int64, site_type; nsweeps=10)
  L = length(vertices(g))
  if (site_type == "Electron" || site_type == "Fermion")
    sites = siteinds(site_type, L; conserve_qns=true)
  else
    sites = siteinds(site_type, L; conserve_qns=false)
  end

  if (site_type == "Electron" || site_type == "S=1/2")
    init_state = [isodd(i) ? "Up" : "Dn" for i in 1:L]
  else
    init_state = [iseven(i) ? "Occ" : "Emp" for i in 1:L]
  end
  ψ0 = randomMPS(sites, init_state)

  sweeps = Sweeps(nsweeps)
  setmaxdim!(sweeps, χ)

  adj_mat = adjacency_matrix(g)
  if (site_type == "S=1/2")
    H = HSpin(adj_mat, params, sites)
  elseif (site_type == "Electron")
    H = Helectron(adj_mat, params, sites)
  elseif (site_type == "Fermion")
    H = Hfermion(adj_mat, params, sites)
  end

  e, ψ = dmrg(H, ψ0, sweeps)

  return println("DMRG Finished and found an energy of " * string(e))
end
