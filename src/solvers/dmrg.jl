using ITensors.ITensorMPS: ITensorMPS, dmrg
using KrylovKit: KrylovKit

"""
Overload of `ITensors.ITensorMPS.dmrg`.
"""

function ITensorMPS.dmrg(
  operator, init_state; nsweeps, nsites=2, updater=eigsolve_updater, kwargs...
)
  return alternating_update(operator, init_state; nsweeps, nsites, updater, kwargs...)
end

"""
Overload of `KrylovKit.eigsolve`.
"""
KrylovKit.eigsolve(H, init::AbstractTTN; kwargs...) = dmrg(H, init; kwargs...)