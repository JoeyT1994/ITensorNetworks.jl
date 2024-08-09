using ITensors: contract
using ITensors.ContractionSequenceOptimization: optimal_contraction_sequence
using KrylovKit: eigsolve, geneigsolve, exponentiate
using LinearAlgebra: norm_sqr
using OMEinsumContractionOrders

function default_krylov_kwargs()
  return (; tol=1e-14, krylovdim=10, maxiter=3, verbosity=0, ishermitian=true, isposdef = true)
end

function get_new_state(
  env_tensors::Vector{ITensor},
  state::ITensor;
  sequence="automatic",
)
  return dag(noprime(contract([copy(state); env_tensors]; sequence)))
end

function bp_eigsolve_updater(
  init::ITensor,
  ∂ψOψ_bpc_∂rs::Vector,
  ∂ψIψ_bpc_∂r::Vector{ITensor},
  krylov_kwargs=default_krylov_kwargs(),
)
  #o_sequences =  [optimal_contraction_sequence([init; ∂ψOψ_bpc_∂r]) for ∂ψOψ_bpc_∂r in ∂ψOψ_bpc_∂rs]
  #n_sequence = optimal_contraction_sequence([init; ∂ψIψ_bpc_∂r])
  n_sequence = contraction_sequence([init; ∂ψIψ_bpc_∂r ]; alg = "sa_bipartite")
  o_sequences =  [contraction_sequence([init; ∂ψOψ_bpc_∂r]; alg = "sa_bipartite") for ∂ψOψ_bpc_∂r in ∂ψOψ_bpc_∂rs]
  H_psi = state -> reduce(+, ITensor[get_new_state(∂ψOψ_bpc_∂r, state; sequence = o_sequence) 
    for (∂ψOψ_bpc_∂r, o_sequence) in zip(∂ψOψ_bpc_∂rs, o_sequences)])
  N_psi = state -> get_new_state(∂ψIψ_bpc_∂r, state; sequence = n_sequence)
  f = state -> (H_psi(state), N_psi(state))
  
  howmany = 1

  vals, vecs, info = geneigsolve(f, init, howmany, :SR; krylov_kwargs...)
  state = first(vecs)
  final_energy = first(vals)

  return state, final_energy
end