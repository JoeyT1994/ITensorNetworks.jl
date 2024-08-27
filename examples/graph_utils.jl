
function named_grid_periodic_x(nxny::Tuple)
    nx, ny = nxny
    g = named_grid((nx,ny))
    for i in 1:ny
        g = add_edge(g, NamedEdge((nx, i) => (1, i)))
    end
    return g
end

function lieb_lattice(nx::Int64, ny::Int64; periodic = false)
    @assert (!periodic && isodd(nx) && isodd(ny)) || (periodic && iseven(nx) && isodd(ny))
    g = named_grid((nx,ny))
    for v in vertices(g)
        if iseven(first(v)) && iseven(last(v))
            g = rem_vertex(g, v)
        end
    end
    return g
end