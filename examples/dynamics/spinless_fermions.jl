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
  full_update_bp,
  get_environment,
  find_subgraph,
  diagblocks,
  initialize_bond_tensors,
  setindex_preserve_graph!,
  simple_update_bp,
  sqrt_and_inv_sqrt,
  simple_update_bp_full,
  spinless_fermions
using Dictionaries
using Observers
using NPZ
using Statistics

using SplitApplyCombine
using Plots

using NamedGraphs: decorate_graph_edges

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

function Hubbard_gates(
  s::IndsNetwork, U; reverse_gates=true, imaginary_time=true, real_time=false, dbeta=-0.2
)
  gates = ITensor[]
  Random.seed!(124435)
  for e in edges(s)
    vsrc, vdst = src(e), dst(e)
    hj =
      -op("Cdag", s[vsrc]) * op("C", s[vdst]) +
      op("C", s[vsrc]) * op("Cdag", s[vdst]) +
      U * op("N", s[vsrc]) * op("N", s[vdst])
    if imaginary_time
      push!(gates, exp_ITensor(hj, dbeta / 2))
    elseif real_time
      if reverse_gates
        push!(gates, exp_ITensor(hj, 1.0 * im * dbeta / 2))
      else
        push!(gates, exp_ITensor(hj, 1.0 * im * dbeta))
      end
    else
      push!(gates, hj)
    end
  end

  if reverse_gates
    append!(gates, reverse(gates))
  end

  return gates
end

function exact_dynamics_hopping_fermionic_model(
  A::Matrix, cidag_cj_init::Matrix, t::Float64
)
  F = eigen(A)
  eigvals, U = F.values, F.vectors
  Udag = adjoint(U)

  ckdag_cl_init = Udag * cidag_cj_init * U
  ckdag_cl_t =
    Diagonal(exp.(eigvals .* t .* im)) * ckdag_cl_init * Diagonal(exp.(-eigvals .* t .* im))

  cidag_cj_t = U * ckdag_cl_t * Udag
  return cidag_cj_t
end

function main(g::NamedGraph, χ::Int64, time_steps::Vector{Float64})
  ITensors.enable_auto_fermion()
  g_vs = vertices(g)
  A = Matrix(adjacency_matrix(g))
  s = siteinds("Fermion", g; conserve_qns=true)

  ψ = ITensorNetwork(s, v -> findfirst(==(v), g_vs) % 2 == 0 ? "Occ" : "Emp")
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])
  mts = message_tensors(
    ψψ;
    subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ)))),
    itensor_constructor=denseblocks ∘ delta,
  )
  mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"))
  init_occs = real.(expect_BP("N", ψ, ψψ, mts; expect_vertices=g_vs))
  cidag_cj_init = Matrix(Diagonal(collect(values(init_occs))))

  time = 0

  println("Starting Sim. Bond dim is $χ")
  for dt in time_steps
    @show time
    u⃗ = Hubbard_gates(
      s, 0.0; dbeta=-dt, imaginary_time=false, real_time=true, reverse_gates=true
    )
    for u in u⃗
      ψ, ψψ, mts = apply(u, ψ, ψψ, mts; maxdim=χ, cutofff=1e-12, simple_BP=true)
    end
    time += dt

    mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"), niters=25)
  end

  final_occs = real.(expect_BP("N", ψ, ψψ, mts; expect_vertices=g_vs))
  cidag_cj = exact_dynamics_hopping_fermionic_model(A, cidag_cj_init, time)
  final_occs_exact = real.(diag(cidag_cj))

  ΔN = abs(sum(values(init_occs)) - sum(values(final_occs)))
  println("Evolution finished. Change in conserved quantity is $ΔN.")

  N_err = mean(abs.(collect(values(final_occs)) - final_occs_exact))
  println("Error on final occs is $N_err")

  return ITensors.disable_auto_fermion()
end

tooth_lengths = fill(6, 6)
g = named_comb_tree(tooth_lengths)
time_steps = [0.05 for i in 1:20]

main(g, 24, time_steps)
