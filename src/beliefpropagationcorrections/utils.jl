using Graphs: AbstractGraph
using NamedGraphs.GraphsExtensions: a_star, rem_edge, neighbors, rem_vertex
using ITensorNetworks: environment, default_message_update, linkinds
using ITensors: ITensors, commonind, delta

function default_krylov_kwargs()
    return (; tol=1e-14, krylovdim=20, maxiter=5, verbosity=0, eager=false, ishermitian=false)
end

function project(ψIψ_bpc::BeliefPropagationCache, pe::PartitionEdge, m_pes::Vector{ITensor}, m_pers::Vector{ITensor})
    ψIψ_bpc = copy(ψIψ_bpc)
    ms = messages(ψIψ_bpc)
    per = reverse(pe)
    me, mer = only(message(ψIψ_bpc, pe)), only(message(ψIψ_bpc, per))
    for (ml, mr) in zip(m_pes, m_pers)
        me = me - me * (mr) * ml
        mer = mer - mer * (ml) * mr
    end
    me /= norm(me)
    mer /= norm(mer)
    set!(ms, pe, ITensor[me])
    set!(ms, per, ITensor[mer])
    return ψIψ_bpc
end 

#Given an updated cache get the next message
function get_eigenvectors(ψIψ_bpc::BeliefPropagationCache, pe::PartitionEdge, howmany::Int64 = 1; maxiter::Int64 = 20)
    per = reverse(pe)
    left_vectors, right_vectors = ITensor[copy(only(message(ψIψ_bpc, per)))], ITensor[copy(only(message(ψIψ_bpc, pe)))]
    #Going parallel seems crucial?!
    pes = [[pe] for pe in default_edge_sequence(ψIψ_bpc)]
    for k in 2:howmany
        ψIψ_bpc = BeliefPropagationCache(partitioned_tensornetwork(ψIψ_bpc))
        ψIψ_bpc = project(ψIψ_bpc, pe, right_vectors, left_vectors)
        for i in 1:maxiter
            ψIψ_bpc = update(ψIψ_bpc, pes)
            ψIψ_bpc = project(ψIψ_bpc, pe, right_vectors, left_vectors)
        end
        n = region_scalar(ψIψ_bpc, pe)
        if n > 0 
            push!(right_vectors, (1 / sqrt(n))*only(message(ψIψ_bpc, pe)))
            push!(left_vectors, (1 / sqrt(n))*only(message(ψIψ_bpc, per)))
        elseif n < 0
            push!(right_vectors, (-1 / sqrt(abs(n)))*only(message(ψIψ_bpc, pe)))
            push!(left_vectors, (1 / sqrt(abs(n)))*only(message(ψIψ_bpc, per)))
        else
            push!(right_vectors, only(message(ψIψ_bpc, pe)))
            push!(left_vectors, only(message(ψIψ_bpc, per)))
        end
    end

    return left_vectors, right_vectors
end

function shortest_path_to_v(g::AbstractGraph, edge, v)
    g_m = rem_edge(g, edge)
    p1 = a_star(g_m, src(edge), v)
    p2 = a_star(g_m, dst(edge), v)
    return p1, p2
end

function get_local_term(qf::QuadraticFormNetwork, v)
    return qf[operator_vertex(qf, v)]*qf[bra_vertex(qf, v)]*qf[ket_vertex(qf, v)]
end

function effective_environment(ψ::ITensorNetwork, ψIψ_bpc::BeliefPropagationCache, v, e, howmany::Int64 = 1; cache_update_kwargs = (;))
    ψIψ = tensornetwork(ψIψ_bpc)
    eigL, eigR = get_eigenvectors(ψIψ_bpc, PartitionEdge(e), howmany; cache_update_kwargs...)
    p1, p2 = shortest_path_to_v(ψ, e, v)
    seq = vcat(PartitionEdge.(p1), PartitionEdge.(p2))
    qf = tensornetwork(ψIψ_bpc)
    ψIψ_bpc_mod = copy(ψIψ_bpc)
    e_region = [bra_vertex(qf, v), ket_vertex(qf, v), operator_vertex(qf, v)]
    environments = Vector{ITensor}[]
    ψIψ_v = ψIψ[operator_vertex(qf, v)]
    s = commonind(ψIψ[ket_vertex(qf, v)], ψIψ_v)
    operator = ITensors.op("Z", s)
    numerator, denominator = 0, 0
    for i in 1:length(eigL)
        ms = messages(ψIψ_bpc_mod)
        set!(ms, PartitionEdge(e), ITensor[eigR[i]])
        set!(ms, reverse(PartitionEdge(e)), ITensor[eigL[i]])
        ψIψ_bpc_mod = update(ψIψ_bpc_mod, seq; message_update = mts -> default_message_update(mts; normalize = false))
        ∂ψIψ_∂v = environment(ψIψ_bpc_mod, [operator_vertex(qf, v)])
        numerator += contract(vcat(∂ψIψ_∂v, operator); sequence = "automatic")[]
        denominator += contract(vcat(∂ψIψ_∂v, ψIψ_v); sequence = "automatic")[] 
    end

    return numerator / denominator
end

function shortest_cycle(g::AbstractGraph, v)
    vn = neighbors(g, v)
    g_m = rem_vertex(g, v)
    paths = [vcat(vcat(NamedEdge(v => vn[i]), a_star(g_m, vn[i], vn[j])), NamedEdge(vn[j] => v)) for i in 1:length(vn) for j in (i+1):length(vn)]
    p = first(sort(paths; by = length))
    return p
end

function get_next_message(ψIψ_bpc::BeliefPropagationCache, seq, init::ITensor)
    ms = messages(ψIψ_bpc)
    e = first(seq)
    set!(ms, e, ITensor[init])
    ψIψ_bpc = update(ψIψ_bpc, seq[2:length(seq)]; message_update = mts -> default_message_update(mts; normalize = false))
    m = only(message(ψIψ_bpc, last(seq)))
    return m
end


function eigendecompose_loop(ψ::ITensorNetwork, ψIψ_bpc::BeliefPropagationCache, v, howmany::Int64 = 1; krylov_kwargs = default_krylov_kwargs())
    loop = shortest_cycle(ψ, v)
    forward_loop, backward_loop = loop, reverse(reverse.(loop))

    pe, per = PartitionEdge(first(forward_loop)), PartitionEdge(first(backward_loop))
    delta_top = delta(only(linkinds(ψ, last(forward_loop))), only(linkinds(ψ, last(backward_loop)))) 
    delta_bottom = dual_index_map(tensornetwork(ψIψ_bpc))(delta_top)

    init = only(default_message(ψIψ_bpc, pe))
    get_new_state_ = state -> (get_next_message(ψIψ_bpc, PartitionEdge.(forward_loop), state)*delta_top)*delta_bottom
    r_vals, r_vecs, info = eigsolve(get_new_state_, init, howmany, :LM; krylov_kwargs...)

    init = only(default_message(ψIψ_bpc, per))
    get_new_state_ = state -> (get_next_message(ψIψ_bpc, PartitionEdge.(backward_loop), state)*delta_top)*delta_bottom
    l_vals, l_vecs, info = eigsolve(get_new_state_, init, howmany, :LM; krylov_kwargs...)

    for i in 1:length(r_vecs)
        n = dot(r_vecs[i], (delta_top)*delta_bottom*l_vecs[i])
        if abs(n) >= 1e-12
            r_vecs[i] /= sqrt(n)
            l_vecs[i] /= sqrt(n)
        end
    end

    l_vecs = [(l_vec * delta_top) * delta_bottom for l_vec in l_vecs]
    r_vecs = [(r_vec * delta_top) * delta_bottom for r_vec in r_vecs]

    return r_vecs, l_vecs, r_vals, l_vals
end


