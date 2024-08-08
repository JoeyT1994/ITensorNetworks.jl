using ITensorNetworks: AbstractITensorNetwork, update_factors, edge_tag, normalize_messages, region_scalar, scalar_factors_quotient, vertex_scalars, environment, factor, tensornetwork
using NamedGraphs.PartitionedGraphs: partitionvertices
using ITensors: ITensor, uniqueinds, factorize_svd, factorize, inds, map_diag!, hascommoninds, OpSum
using LinearAlgebra: norm, ishermitian
using Dictionaries: Dictionary

function renormalize_update_norm_cache(
    ψ::ITensorNetwork,
    ψIψ_bpc::BeliefPropagationCache;
    cache_update_kwargs,
    update_cache = true,
  )

    ψ = copy(ψ)
    if update_cache
      ψIψ_bpc = update(ψIψ_bpc; cache_update_kwargs...)
    end
    ψIψ_bpc = normalize_messages(ψIψ_bpc)
    qf = tensornetwork(ψIψ_bpc)
  
    for v in vertices(ψ)
      v_ket, v_bra = ket_vertex(qf, v), bra_vertex(qf, v)
      pv = only(partitionvertices(ψIψ_bpc, [v_ket]))
      vn = region_scalar(ψIψ_bpc, pv)
      state = copy(ψ[v]) / sqrt(vn)
      state_dag = copy(dag(state))
      state_dag = replaceinds(
        state_dag, inds(state_dag), dual_index_map(qf).(inds(state_dag))
      )
      vertices_states = Dictionary([v_ket, v_bra], [state, state_dag])
      ψIψ_bpc = update_factors(ψIψ_bpc, vertices_states)
      ψ[v] = state
    end

    return ψ, ψIψ_bpc
end

#TODO: Add support for nsites = 2
function bp_inserter(
    ψ::AbstractITensorNetwork,
    ψIψ_bpc::BeliefPropagationCache,
    state::ITensor,
    region;
    nsites::Int64=1,
    bp_update_kwargs,
    kwargs...,
  )
    ψ = copy(ψ)
    form_network = tensornetwork(ψIψ_bpc)
    if length(region) == 1
      states = ITensor[state]
    else
      error("Region lengths of more than 1 not supported for now")
    end
  
    for (state, v) in zip(states, region)
      ψ[v] = state
      state_dag = dag(copy(ψ[v]))
      state_dag = replaceinds(
        state_dag, inds(state_dag), dual_index_map(form_network).(inds(state_dag))
      )
      form_bra_v, form_ket_v = bra_vertex(form_network, v), ket_vertex(form_network, v)
      vertices_states = Dictionary([form_ket_v, form_bra_v], [state, state_dag])
      ψIψ_bpc = update_factors(ψIψ_bpc, vertices_states)
    end
  
    ψ, ψIψ_bpc = renormalize_update_norm_cache(
      ψ, ψIψ_bpc; cache_update_kwargs=bp_update_kwargs
    )
  
    return ψ, ψIψ_bpc
  end
  
