using Compat: Compat
using Graphs: vertices
# Trigger package extension.
using ITensorNetworks:
  ITensorNetworks,
  BeliefPropagationCache,
  ⊗,
  combine_linkinds,
  contract,
  contract_boundary_mps,
  contraction_sequence,
  eachtensor,
  environment,
  inner_network,
  linkinds_combiners,
  message,
  partitioned_tensornetwork,
  random_tensornetwork,
  siteinds,
  split_index,
  tensornetwork,
  update,
  update_factor,
  update_message,
  QuadraticFormNetwork,
  messages
using ITensors: ITensors, ITensor, combiner, dag, inds, inner, op, prime, randomITensor
using ITensorNetworks.ModelNetworks: ModelNetworks
using ITensors.NDTensors: array
using LinearAlgebra: eigvals, tr
using NamedGraphs: NamedEdge
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid
using NamedGraphs.PartitionedGraphs: PartitionVertex, partitionedges
using Random: Random
using SplitApplyCombine: group
using Suppressor

ITensors.disable_warn_order()

function main()
  g = named_grid((10, 10))
  s = siteinds("S=1/2", g)
  χ = 10
  Random.seed!(1234)
  ψ = random_tensornetwork(s; link_space=χ)
  bpc = BeliefPropagationCache(QuadraticFormNetwork(ψ))
  edge = first(partitionedges(partitioned_tensornetwork(bpc)))
  @time bpc = update(bpc; maxiter=20, tol=1e-10, verbose=true)
end

main()
