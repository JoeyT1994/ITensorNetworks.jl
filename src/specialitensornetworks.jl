using ITensors: delta
using ITensors.NDTensors: dim
using DataGraphs: IsUnderlyingGraph
using Distributions: Distribution

"""
RETURN A TENSOR NETWORK WITH COPY TENSORS ON EACH VERTEX. 
Note that passing a link_space will mean the indices of the resulting network don't match those of the input indsnetwork
"""
function delta_network(eltype::Type, s::IndsNetwork; kwargs...)
  return ITensorNetwork(s; kwargs...) do v
    return inds -> delta(eltype, inds)
  end
end

function delta_network(s::IndsNetwork; kwargs...)
  return delta_network(Float64, s; kwargs...)
end

function delta_network(eltype::Type, graph::AbstractNamedGraph; kwargs...)
  return delta_network(eltype, IndsNetwork(graph; kwargs...))
end

function delta_network(graph::AbstractNamedGraph; kwargs...)
  return delta_network(Float64, graph; kwargs...)
end

"""
Build an ITensor network on a graph specified by the inds network s. Bond_dim is given by link_space and entries are randomised (normal distribution, mean 0 std 1)
"""
function random_tensornetwork(eltype::Type, s::IndsNetwork; kwargs...)
  return ITensorNetwork(s; kwargs...) do v
    return inds -> itensor(randn(eltype, dim.(inds)...), inds)
  end
end

function random_tensornetwork(s::IndsNetwork; kwargs...)
  return random_tensornetwork(Float64, s; kwargs...)
end

@traitfn function random_tensornetwork(eltype::Type, g::::IsUnderlyingGraph; kwargs...)
  return random_tensornetwork(eltype, IndsNetwork(g); kwargs...)
end

@traitfn function random_tensornetwork(g::::IsUnderlyingGraph; kwargs...)
  return random_tensornetwork(Float64, IndsNetwork(g); kwargs...)
end

"""
Build an ITensor network on a graph specified by the inds network s.
Bond_dim is given by link_space and entries are randomized.
The random distribution is based on the input argument `distribution`.
"""
function random_tensornetwork(distribution::Distribution, s::IndsNetwork; kwargs...)
  return ITensorNetwork(s; kwargs...) do v
    return inds -> itensor(rand(distribution, dim.(inds)...), inds)
  end
end

@traitfn function random_tensornetwork(
  distribution::Distribution, g::::IsUnderlyingGraph; kwargs...
)
  return random_tensornetwork(distribution, IndsNetwork(g); kwargs...)
end
