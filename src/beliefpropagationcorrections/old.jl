using NamedGraphs.GraphsExtensions: vertices, src, dst, rem_edges
using NamedGraphs: NamedEdge
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs.PartitionedGraphs: PartitionEdge, partitionedges
using ITensors: ITensor, siteinds, contract, inds, prime, mapprime
using ITensors.NDTensors: svd, eigen, array
using ITensorNetworks: ITensorNetwork, random_tensornetwork, QuadraticFormNetwork, bra_vertex, ket_vertex, operator_vertex, combine_linkinds, split_index, BeliefPropagationCache, update, message, partitioned_tensornetwork, messages, region_scalar,
    normalize_messages
using LinearAlgebra: norm, eigvals, dot, diag
using Dictionaries: set!

using Random

Random.seed!(1437)

function get_local_term(qf, v)
    return qf[operator_vertex(qf, v)]*qf[bra_vertex(qf, v)]*qf[ket_vertex(qf, v)]
end


L = 4
g = named_grid((L,1); periodic = true)
s = siteinds("S=1/2", g)
ψ = random_tensornetwork(s; link_space = 2)
qf = QuadraticFormNetwork(ψ)

ψψ = copy(ψ)
for v in vertices(ψ)
    ψψ[v] = get_local_term(qf, v)
end

function project(ψ::ITensorNetwork, e, me::Vector{ITensor}, mer::Vector{ITensor})
    ψ = copy(ψ)
    projψ = sum([(copy(ψ[src(e)]) * mer[i]) * me[i] for i in 1:length(me)])
    ψ[src(e)] = ψ[src(e)] - projψ

    projψ = sum([(copy(ψ[dst(e)]) * me[i]) * mer[i] for i in 1:length(me)])
    ψ[dst(e)] = ψ[dst(e)] - projψ

    return ψ
end


ψψ = combine_linkinds(ψψ)

edge_to_split = NamedEdge((2,1) => (3,1))

ψψsplit = split_index(ψψ, [edge_to_split])
edge_env = contract(ψψsplit)

bpc = BeliefPropagationCache(ψψ)
bpc = update(bpc; maxiter = 30)
bpc = normalize_messages(bpc)
me1, mer1 = only(message(bpc, PartitionEdge(edge_to_split))), only(message(bpc, reverse(PartitionEdge(edge_to_split))))

eigs = reverse(sort(eigvals(array(edge_env)); by = abs))
eigmax = first(eigs)
eigsecondmax = eigs[2]
eigthirdmax = eigs[3]
eigfourthmax = eigs[4]
D, U = eigen(edge_env)

ψψsplit = project(ψψ, edge_to_split, [me1], [mer1])
bpc = BeliefPropagationCache(ψψsplit)
bpc = update(bpc; maxiter = 30)
me2, mer2 = only(message(bpc, PartitionEdge(edge_to_split))), only(message(bpc, reverse(PartitionEdge(edge_to_split))))
n = dot(me2, mer2)
if n < 0
    me2 *= -1
end
me2, mer2 = me2 / sqrt(abs(n)), mer2 / sqrt(abs(n))


ψψsplit = project(ψψ, edge_to_split, [me1, me2], [mer1, mer2])
bpc = BeliefPropagationCache(ψψsplit)
bpc = update(bpc; maxiter = 30)
me3, mer3 = only(message(bpc, PartitionEdge(edge_to_split))), only(message(bpc, reverse(PartitionEdge(edge_to_split))))
n = dot(me3, mer3)
if n < 0
    me3 *= -1
end
me3, mer3 = me3 / sqrt(abs(n)), mer3 / sqrt(abs(n))

ψψsplit = project(ψψ, edge_to_split, [me1, me2, me3], [mer1, mer2, mer3])
bpc = BeliefPropagationCache(ψψsplit)
bpc = update(bpc; maxiter = 30)
me4, mer4 = only(message(bpc, PartitionEdge(edge_to_split))), only(message(bpc, reverse(PartitionEdge(edge_to_split))))
n = dot(me4, mer4)
if n < 0
    me4 *= -1
end
me4, mer4 = me4 / sqrt(abs(n)), mer4 / sqrt(abs(n))

edge_env_approx1 = eigmax*(me1) * prime(mer1)
edge_env_approx2 = edge_env_approx1 + eigsecondmax*(me2) * prime(mer2)
edge_env_approx3 = edge_env_approx2 + eigfourthmax * (me3) * prime(mer3)
edge_env_approx4 = edge_env_approx3 + eigthirdmax * (me4) * prime(mer4)

@show edge_env
@show edge_env_approx4

#@show edge_env
#@show edge_env_approx4

@show norm(edge_env - edge_env_approx1)
@show norm(edge_env - edge_env_approx2)
@show norm(edge_env - edge_env_approx3)
@show norm(edge_env - edge_env_approx4)