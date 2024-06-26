using ITensors: ITensor, contract, dag, onehot, uniqueind
using ITensors.NDTensors: array
using LinearAlgebra: eigen

function dmrg_x_updater(
  init;
  state!,
  projected_operator!,
  outputlevel,
  which_sweep,
  sweep_plan,
  which_region_update,
  internal_kwargs,
)
  H = contract(projected_operator![], ITensor(true))
  D, U = eigen(H; ishermitian=true)
  u = uniqueind(U, H)
  max_overlap, max_ind = findmax(abs, array(dag(init) * U))
  U_max = U * dag(onehot(u => max_ind))
  eigvals = [((onehot(u => max_ind)' * D) * dag(onehot(u => max_ind)))[]]
  return U_max, (; eigvals)
end
