using Dictionaries: Dictionary
using Graphs: AbstractGraph, nv, vertices
using ITensors: ITensors, Index, siteind, siteinds

function ITensors.siteind(sitetype::String, v::Tuple; kwargs...)
  return addtags(siteind(sitetype; kwargs...), vertex_tag(v))
end

# naming collision of ITensors.addtags and addtags keyword in siteind system
function ITensors.siteind(d::Integer, v; addtags="", kwargs...)
  return ITensors.addtags(Index(d; tags="Site, $addtags", kwargs...), vertex_tag(v))
end

function ITensors.siteinds(sitetypes::AbstractDictionary, g::AbstractGraph; kwargs...)
  is = IndsNetwork(g)
  for v in vertices(g)
    is[v] = [siteind(sitetypes[v], vertex_tag(v); kwargs...)]
  end
  return is
end

function ITensors.siteinds(sitetype, g::AbstractGraph; kwargs...)
  return siteinds(Dictionary(vertices(g), fill(sitetype, nv(g))), g; kwargs...)
end

function ITensors.siteinds(f::Function, g::AbstractGraph; kwargs...)
  return siteinds(Dictionary(vertices(g), map(v -> f(v), vertices(g))), g; kwargs...)
end
