module ModelHamiltonians
using Dictionaries: AbstractDictionary
using Graphs: AbstractGraph, dst, edges, edgetype, neighborhood, path_graph, src, vertices
using ITensors.Ops: OpSum

to_callable(value::Type) = value
to_callable(value::Function) = value
to_callable(value::AbstractDict) = Base.Fix1(getindex, value)
to_callable(value::AbstractDictionary) = Base.Fix1(getindex, value)
function to_callable(value::AbstractArray{<:Any,N}) where {N}
  getindex_value(x::Integer) = value[x]
  getindex_value(x::Tuple{Vararg{Integer,N}}) = value[x...]
  getindex_value(x::CartesianIndex{N}) = value[x]
  return getindex_value
end
to_callable(value) = Returns(value)

# TODO: Consider removing the if-statements checking !iszero. They
# help to avoid constructing non-nearest
# neighbor gates, which `apply` can't handle
# right now. Maybe we could skip zero terms in gate
# construction or application instead, but maybe it should be handled here.

function tight_binding(g::AbstractGraph; t=1, tp=0, h=0)
  (; t, tp, h) = map(to_callable, (; t, tp, h))
  h = to_callable(h)
  ℋ = OpSum()
  for e in edges(g)
    if !iszero(t(e))
      ℋ -= t(e), "Cdag", src(e), "C", dst(e)
      ℋ -= t(e), "Cdag", dst(e), "C", src(e)
    end
  end
  for v in vertices(g)
    for nn in next_nearest_neighbors(g, v)
      e = edgetype(g)(v, nn)
      if !iszero(tp(e))
        ℋ -= tp(e), "Cdag", src(e), "C", dst(e)
        ℋ -= tp(e), "Cdag", dst(e), "C", src(e)
      end
    end
  end
  for v in vertices(g)
    if !iszero(h(v))
      ℋ -= h(v), "N", v
    end
  end
  return ℋ
end

"""
t-t' Hubbard Model g,i,v
"""
function hubbard(g::AbstractGraph; U=0, t=1, tp=0, h=0)
  (; U, t, tp, h) = map(to_callable, (; U, t, tp, h))
  ℋ = OpSum()
  for e in edges(g)
    if !iszero(t(e))
      ℋ -= t(e), "Cdagup", src(e), "Cup", dst(e)
      ℋ -= t(e), "Cdagup", dst(e), "Cup", src(e)
      ℋ -= t(e), "Cdagdn", src(e), "Cdn", dst(e)
      ℋ -= t(e), "Cdagdn", dst(e), "Cdn", src(e)
    end
  end
  for v in vertices(g)
    for nn in next_nearest_neighbors(g, v)
      e = edgetype(g)(v, nn)
      if !iszero(tp(e))
        ℋ -= tp(e), "Cdagup", src(e), "Cup", dst(e)
        ℋ -= tp(e), "Cdagup", dst(e), "Cup", src(e)
        ℋ -= tp(e), "Cdagdn", src(e), "Cdn", dst(e)
        ℋ -= tp(e), "Cdagdn", dst(e), "Cdn", src(e)
      end
    end
  end
  for v in vertices(g)
    if !iszero(h(v))
      ℋ -= h(v), "Sz", v
    end
    if !iszero(U(v))
      ℋ += U(v), "Nupdn", v
    end
  end
  return ℋ
end

"""
Random field J1-J2 Heisenberg model on a general graph
"""
function heisenberg(g::AbstractGraph; J1=1, J2=0, h=0)
  (; J1, J2, h) = map(to_callable, (; J1, J2, h))
  ℋ = OpSum()
  for e in edges(g)
    if !iszero(J1(e))
      ℋ += J1(e) / 2, "S+", src(e), "S-", dst(e)
      ℋ += J1(e) / 2, "S-", src(e), "S+", dst(e)
      ℋ += J1(e), "Sz", src(e), "Sz", dst(e)
    end
  end
  for v in vertices(g)
    for nn in next_nearest_neighbors(g, v)
      e = edgetype(g)(v, nn)
      if !iszero(J2(e))
        ℋ += J2(e) / 2, "S+", src(e), "S-", dst(e)
        ℋ += J2(e) / 2, "S-", src(e), "S+", dst(e)
        ℋ += J2(e), "Sz", src(e), "Sz", dst(e)
      end
    end
  end
  for v in vertices(g)
    if !iszero(h(v))
      ℋ += h(v), "Sz", v
    end
  end
  return ℋ
end

"""
Random field J1-J2 Heisenberg model on a chain of length N
"""
heisenberg(N::Integer; kwargs...) = heisenberg(path_graph(N); kwargs...)

"""
Next-to-nearest-neighbor Ising model (ZZX) on a general graph
"""
function ising(g::AbstractGraph; J1=-1, J2=0, h=0)
  (; J1, J2, h) = map(to_callable, (; J1, J2, h))
  ℋ = OpSum()
  for e in edges(g)
    if !iszero(J1(e))
      ℋ += J1(e), "Sz", src(e), "Sz", dst(e)
    end
  end
  for v in vertices(g)
    for nn in next_nearest_neighbors(g, v)
      e = edgetype(g)(v, nn)
      if !iszero(J2(e))
        ℋ += J2(e), "Sz", src(e), "Sz", dst(e)
      end
    end
  end
  for v in vertices(g)
    if !iszero(h(v))
      ℋ += h(v), "Sx", v
    end
  end
  return ℋ
end

"""
Next-to-nearest-neighbor Ising model (ZZX) on a chain of length N
"""
ising(N::Integer; kwargs...) = ising(path_graph(N); kwargs...)
end
