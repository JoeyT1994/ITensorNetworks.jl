using ITensors
using NamedGraphs
using Graphs

using ITensorTDVP

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

function HFermion(adj_mat, params, sites)
  ampo = OpSum()
  @assert size(adj_mat)[1] == size(adj_mat)[2]
  L = size(adj_mat)[1]
  for i in 1:L
    for j in (i + 1):L
      ampo += params["t"] * adj_mat[i, j], "Cdag", i, "C", j
      ampo += params["t"] * adj_mat[i, j], "Cdag", j, "C", i
      ampo += params["U"] * adj_mat[i, j], "N", j, "N", i
    end
  end

  H = MPO(ampo, sites)
  return H
end

function HSpinQN(adj_mat, params, sites)
  ampo = OpSum()
  @assert size(adj_mat)[1] == size(adj_mat)[2]
  L = size(adj_mat)[1]
  for i in 1:L
    for j in (i + 1):L
      ampo += 2 * adj_mat[i, j] * params["J"], "S+", i, "S-", j
      ampo += 2 * adj_mat[i, j] * params["J"], "S-", i, "S+", j
      ampo += adj_mat[i, j] * params["Δ"], "Z", i, "Z", j
    end
  end

  for i in 1:L
    if haskey(params, "hz")
      ampo += params["hz"], "Z", i
    end
  end
  H = MPO(ampo, sites)
  return H
end

function spin_DMRG_backend(params, g::AbstractGraph, χ::Int64; nsweeps=20)
  L = length(vertices(g))
  adj_mat = adjacency_matrix(g)

  if haskey(params, "J") && !haskey(params, "Jx") && !haskey(params, "Jy")
    sites = siteinds("S=1/2", L; conserve_qns=true)
    H = HSpinQN(adj_mat, params, sites)
  else
    sites = siteinds("S=1/2", L; conserve_qns=false)
    H = HSpin(adj_mat, params, sites)
  end

  init_state = [isodd(i) ? "Up" : "Dn" for i in 1:L]
  ψ0 = randomMPS(sites, init_state)

  sweeps = Sweeps(nsweeps)
  setmaxdim!(sweeps, χ)

  e, ψ = dmrg(H, ψ0, sweeps)

  return println("DMRG Finished and found an energy of " * string(e))
end

function fermion_DMRG_backend(params, g::AbstractGraph, χ::Int64; nsweeps=20)
  L = length(vertices(g))
  sites = siteinds("Fermion", L; conserve_nf=true, conserve_sz=true)
  init_state = [isodd(i) ? "Occ" : "Emp" for i in 1:L]
  ψ0 = randomMPS(sites, init_state; linkdims=10)

  sweeps = Sweeps(nsweeps)
  setmaxdim!(sweeps, χ)

  adj_mat = adjacency_matrix(g)
  H = HFermion(adj_mat, params, sites)

  e, ψ = dmrg(H, ψ0, sweeps)

  return println("DMRG Finished and found an energy of " * string(e))
end

function spin_TDVP_backend(
  params, g::AbstractGraph, χ::Int64, tfinal::Float64, delta_t::Float64
)
  g_vs = vertices(g)
  L = length(g_vs)
  adj_mat = adjacency_matrix(g)

  if haskey(params, "J") && !haskey(params, "Jx") && !haskey(params, "Jy")
    sites = siteinds("S=1/2", L; conserve_qns=true)
    H = HSpinQN(adj_mat, params, sites)
  else
    sites = siteinds("S=1/2", L; conserve_qns=false)
    H = HSpin(adj_mat, params, sites)
  end

  init_state = [findfirst(==(v), g_vs) % 2 == 0 ? "Dn" : "Up" for v in g_vs]
  ψ0 = randomMPS(sites, init_state; linkdims=1)

  e_init = inner(ψ0', H, ψ0) / inner(ψ0, ψ0)
  ψfinal = ITensorTDVP.tdvp(
    H,
    -im * tfinal,
    ψ0;
    time_step=-im * delta_t,
    normalize=true,
    maxdim=χ,
    cutoff=1e-12,
    outputlevel=1,
  )
  e_final = inner(ψfinal', H, ψfinal) / inner(ψfinal, ψfinal)
  println("TDVP finished. Final energy is $e_final. Initial energy was $e_init.")
  init_mags = expect(ψ0, "Z")
  final_mags = expect(ψfinal, "Z")
  @show init_mags
  @show final_mags
end

function fermion_TDVP_backend(
  params, g::AbstractGraph, χ::Int64, tfinal::Float64, delta_t::Float64
)
  g_vs = vertices(g)
  L = length(g_vs)
  adj_mat = adjacency_matrix(g)

  sites = siteinds("Fermion", L; conserve_nf=true, conserve_sz=true)
  H = HFermion(adj_mat, params, sites)

  init_state = [findfirst(==(v), g_vs) % 2 == 0 ? "Occ" : "Emp" for v in g_vs]
  ψ0 = randomMPS(sites, init_state; linkdims=1)

  e_init = inner(ψ0', H, ψ0) / inner(ψ0, ψ0)
  ψfinal = ITensorTDVP.tdvp(
    H,
    -im * tfinal,
    ψ0;
    time_step=-im * delta_t,
    normalize=true,
    maxdim=χ,
    cutoff=1e-12,
    outputlevel=1,
  )
  e_final = inner(ψfinal', H, ψfinal) / inner(ψfinal, ψfinal)
  println("TDVP finished. Final energy is $e_final. Initial energy was $e_init.")
  init_occs = expect(ψ0, "N")
  final_occs = expect(ψfinal, "N")
  @show init_occs
  @show final_occs
end
