using NamedGraphs.GraphsExtensions: vertices, src, dst
using NamedGraphs: NamedEdge
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs.PartitionedGraphs: PartitionEdge
using ITensors: siteinds, contract, inds, prime, mapprime
using ITensors.NDTensors: svd, eigen, array
using ITensorNetworks:
  ITensorNetwork,
  random_tensornetwork,
  QuadraticFormNetwork,
  bra_vertex,
  ket_vertex,
  operator_vertex,
  combine_linkinds,
  split_index,
  BeliefPropagationCache,
  update,
  message
using LinearAlgebra: norm, eigvals, dot

using Random

Random.seed!(1437)

function get_local_term(qf, v)
  return qf[operator_vertex(qf, v)] * qf[bra_vertex(qf, v)] * qf[ket_vertex(qf, v)]
end

L = 6
g = named_grid((L, 1); periodic=true)
s = siteinds("S=1/2", g)
ψ = random_tensornetwork(s; link_space=3)
qf = QuadraticFormNetwork(ψ)

ψψ = copy(ψ)
for v in vertices(ψ)
  ψψ[v] = get_local_term(qf, v)
end

function project(ψ::ITensorNetwork, e, me, mer)
  ψ = copy(ψ)
  projψ = (copy(ψ[src(e)]) * mer) * me
  ψ[src(e)] = ψ[src(e)] - projψ

  projψ = (copy(ψ[dst(e)]) * me) * mer
  ψ[dst(e)] = ψ[dst(e)] - projψ

  return ψ
end

ψψ = combine_linkinds(ψψ)

edge_to_split = NamedEdge((1, 1) => (2, 1))

ψψsplit = split_index(ψψ, [edge_to_split])
edge_env = contract(ψψsplit)
#edge_env_sq = mapprime(edge_env * prime(edge_env), 2=>1)
edge_env_sq = edge_env

bpc = BeliefPropagationCache(ψψ)
bpc = update(bpc; maxiter=25)

me1, mer1 = only(message(bpc, PartitionEdge(edge_to_split))),
only(message(bpc, reverse(PartitionEdge(edge_to_split))))
@show me1, mer1

U, D = svd(edge_env_sq, first(inds(edge_env_sq)))

@show U, S, V

ψψ = project(ψψ, edge_to_split, me1, mer1)

ψψsplit = split_index(ψψ, [edge_to_split])
edge_env = contract(ψψsplit)
edge_env_sq = mapprime(edge_env * prime(edge_env), 2 => 1)
U, S, V = svd(edge_env_sq, first(inds(edge_env_sq)))

#@show U, S, V

bpc = BeliefPropagationCache(ψψ)
bpc = update(bpc; maxiter=20)
me2, mer2 = only(message(bpc, PartitionEdge(edge_to_split))),
only(message(bpc, reverse(PartitionEdge(edge_to_split))))
#@show me2, mer2
