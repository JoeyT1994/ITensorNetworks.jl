using ITensors: contract
using ITensors.ContractionSequenceOptimization: optimal_contraction_sequence
using KrylovKit: eigsolve, geneigsolve, exponentiate
using LinearAlgebra: norm_sqr

function default_krylov_kwargs()
  return (; tol=1e-14, krylovdim=20, maxiter=5, verbosity=0, eager=false, ishermitian=true)
end

#TODO: Put inv_sqrt_mts onto ∂ψOψ_bpc_∂r beforehand. Need to do this in an efficient way without
#precontracting ∂ψOψ_bpc_∂r and getting the index logic too messy
function get_new_state(
  ∂ψOψ_bpc_∂rs::Vector,
  inv_sqrt_mts,
  sqrt_mts,
  state::ITensor;
  sequences=["automatic" for i in length(∂ψOψ_bpc_∂rs)],
)
  state = noprime(contract([state; inv_sqrt_mts]))
  states = ITensor[
    noprime(contract([copy(state); ∂ψOψ_bpc_∂r]; sequence)) for
    (∂ψOψ_bpc_∂r, sequence) in zip(∂ψOψ_bpc_∂rs, sequences)
  ]
  state = reduce(+, states)
  return noprime(contract([state; (inv_sqrt_mts)]))
end

function bp_eigsolve_updater(
  init::ITensor,
  ∂ψOψ_bpc_∂rs::Vector,
  sqrt_mts,
  inv_sqrt_mts;
  krylov_kwargs=default_krylov_kwargs(),
)
  gauged_init = noprime(contract([copy(init); sqrt_mts]))
  gauged_init /= norm(gauged_init)
  sequences =  [optimal_contraction_sequence([gauged_init; ∂ψOψ_bpc_∂r]) for ∂ψOψ_bpc_∂r in ∂ψOψ_bpc_∂rs]
  get_new_state_ =
    state -> get_new_state(∂ψOψ_bpc_∂rs, inv_sqrt_mts, sqrt_mts, state; sequences)
  howmany = 1

  vals, vecs, info = eigsolve(get_new_state_, gauged_init, howmany, :SR; krylov_kwargs...)
  state = first(vecs)
  final_energy = first(vals)
  state = noprime(contract([state; inv_sqrt_mts]))

  return state, final_energy
end

function bp_eigsolve_updater_V2(
  init::ITensor,
  ∂ψOψ_bpc_∂rs::Vector,
  sqrt_mts,
  inv_sqrt_mts,
  prev_energy::Float64,
  ψ::ITensorNetwork,
  ψIψ_bpc::BeliefPropagationCache,
  H::OpSum,
  region;
  krylov_kwargs=default_krylov_kwargs(),
  bp_update_kwargs= (;),
  inserter_kwargs = (;)
)
  gauged_init = noprime(contract([copy(init); sqrt_mts]))
  gauged_init /= norm(gauged_init)
  sequences =  [optimal_contraction_sequence([gauged_init; ∂ψOψ_bpc_∂r]) for ∂ψOψ_bpc_∂r in ∂ψOψ_bpc_∂rs]
  get_new_state_ =
    state -> get_new_state(∂ψOψ_bpc_∂rs, inv_sqrt_mts, sqrt_mts, state; sequences)
  howmany = 1

  vals, vecs, info = eigsolve(get_new_state_, gauged_init, 1, :SR; krylov_kwargs...)
  new_state = first(vecs)
  new_state = noprime(contract([new_state; inv_sqrt_mts]))
  ψ_new, ψIψ_bpc_new = bp_inserter(ψ, ψIψ_bpc, new_state, sqrt_mts, inv_sqrt_mts, region; bp_update_kwargs, inserter_kwargs...)
  new_energy = real(sum(expect(ψ_new, H; alg="bp", (cache!)=Ref(ψIψ_bpc_new)))) / L
  if new_energy < prev_energy
    return ψ_new, ψIψ_bpc_new, new_energy
  end

  t = 5
  niters = 10
  for i in 1:niters
    state, info = exponentiate(get_new_state_, -t, gauged_init; krylov_kwargs...)
    state = state / norm(state)
    state = noprime(contract([state; inv_sqrt_mts]))
    ψ_new, ψIψ_bpc_new = bp_inserter(ψ, ψIψ_bpc, state, sqrt_mts, inv_sqrt_mts, region; bp_update_kwargs, inserter_kwargs...)
    new_energy = real(sum(expect(ψ_new, H; alg="bp", (cache!)=Ref(ψIψ_bpc_new)))) / L
    if new_energy < prev_energy
      println("Solution found on iter $i")
      return ψ_new, ψIψ_bpc_new, new_energy
    else
      t *= 0.5
    end
  end

  println("Solution not found!")
  ψ_new, ψIψ_bpc_new = bp_inserter(ψ, ψIψ_bpc, init, sqrt_mts, inv_sqrt_mts, region; bp_update_kwargs, inserter_kwargs...)
  return ψ_new, ψIψ_bpc_new, prev_energy
end