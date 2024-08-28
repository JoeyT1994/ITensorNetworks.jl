
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

function grid_inscribed_hexagonal_lattice_graph(nx::Int64, ny::Int64)
    g = named_hexagonal_lattice_graph(nx, ny)
    g = rename_vertices(v -> (last(v), first(v)), g)
    return g
end

function heavy_hexagonal_lattice_graph(nx::Int64, ny::Int64)
    g = named_hexagonal_lattice_graph(nx, ny)
    g = rename_vertices(v -> (2*first(v), 2*last(v)), g)
    for e in edges(g)
        vsrc, vdst = src(e), dst(e)
        v_new = ((first(vsrc) + first(vdst))/ 2, (last(vsrc) + last(vdst))/ 2)
        g = add_vertex(g, v_new)
        g = rem_edge(g, e)
        g = add_edges(g, [NamedEdge(vsrc => v_new), NamedEdge(v_new => vdst)])
    end
    return g
end