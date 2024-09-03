
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

function ibm_processor_grid()
    g = named_grid((15,13))
    for row in [2,6,10]
        for col in [1,2,4,5,6,8,9,10,12,13,14]
            g = rem_vertex(g, (col, row))
        end
    end

    for row in [4,8,12]
        for col in [2,3,4,6,7,8,10,11,12,14,15]
            g = rem_vertex(g, (col, row))
        end
    end

    g = rem_vertex(g, (1,1))
    g = rem_vertex(g, (15,13))
    return g
end

function ibm_processor_grid_periodic_x()
    g = named_grid((16,13))
    for row in [2,6,10]
        for col in [1,2,4,5,6,8,9,10,12,13,14, 16]
            g = rem_vertex(g, (col, row))
        end
    end

    for row in [4,8,12]
        for col in [2,3,4,6,7,8,10,11,12,14,15, 16]
            g = rem_vertex(g, (col, row))
        end
    end

    for v in filter(v -> first(v) == 16, collect(vertices(g)))
        g = add_edge(g, NamedEdge(v => (1, last(v))))
    end

    return g
end