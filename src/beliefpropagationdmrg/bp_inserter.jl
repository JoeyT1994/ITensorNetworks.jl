using ITensorNetworks: update_factors, edge_tag, renormalize_messages, region_scalar, scalar_factors_quotient, vertex_scalars, environment
using ITensors: ITensor, uniqueinds, factorize_svd, factorize, inds, map_diag!, hascommoninds
using LinearAlgebra: norm
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
    ψIψ_bpc = renormalize_messages(ψIψ_bpc)
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
    sqrt_mts,
    inv_sqrt_mts,
    region;
    nsites::Int64=1,
    bp_update_kwargs,
    kwargs...,
  )
    ψ = copy(ψ)
    form_network = tensornetwork(ψIψ_bpc)
    if length(region) == 1
      states = ITensor[state]
    elseif length(region) == 2
      state = noprime(contract([state; sqrt_mts]))
      v1, v2 = first(region), last(region)
      e = NamedEdge(v1 => v2)
      pe = PartitionEdge(e)
      singular_values! = Ref(ITensor())
      state_v1, state_v2= factorize_svd(state, uniqueinds(ψ[v1], ψ[v2]); singular_values!, tags=edge_tag(e), kwargs...)
      inv_sqrt_mts_v1 = filter(mt -> hascommoninds(mt, state_v1), inv_sqrt_mts)
      inv_sqrt_mts_v2 = filter(mt -> hascommoninds(mt, state_v2), inv_sqrt_mts)
      state_v1 = noprime(contract([state_v1; inv_sqrt_mts_v1]))
      state_v2 = noprime(contract([state_v2; inv_sqrt_mts_v2]))
      states = ITensor[state_v1, state_v2]
      ind2 = commonind(singular_values![], state_v1)
      δuv = dag(copy(singular_values![]))
      δuv = replaceind(δuv, ind2, ind2')
      map_diag!(sign, δuv, δuv)
      singular_values![] = denseblocks(singular_values![]) * denseblocks(δuv)
      mts = messages(ψIψ_bpc)
      set!(mts, pe, dag.(ITensor[singular_values![]]))
      set!(mts, reverse(pe), ITensor[singular_values![]])
    else
      error("Region lengths of more than 2 not supported for now")
    end
  
    for (state, v) in zip(states, region)
      ψ[v] = state
      state_dag = copy(ψ[v])
      state_dag = replaceinds(
        dag(state_dag), inds(state_dag), dual_index_map(form_network).(inds(state_dag))
      )
      form_bra_v, form_op_v, form_ket_v = bra_vertex(form_network, v),
      operator_vertex(form_network, v),
      ket_vertex(form_network, v)
      vertices_states = Dictionary([form_ket_v, form_bra_v], [state, state_dag])
      ψIψ_bpc = update_factors(ψIψ_bpc, vertices_states)
    end
  
    ψ, ψIψ_bpc = renormalize_update_norm_cache(
      ψ, ψIψ_bpc; cache_update_kwargs=bp_update_kwargs
    )
  
    return ψ, ψIψ_bpc
  end
  
