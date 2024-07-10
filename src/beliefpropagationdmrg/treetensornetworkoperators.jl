using NamedGraphs: edges
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs.GraphsExtensions: forest_cover, default_root_vertex, vertices, dfs_tree, undirected_graph,
    bfs_tree
using ITensorNetworks: IndsNetwork

function opsum_to_edge_term_dict(s::IndsNetwork, H::OpSum)
    es = edges(s)
    term_dict = Dictionary(edges(s), [OpSum() for e in edges(s)])
    for (i, term) in enumerate(H)
        if length(sites(term)) == 1 && !iszero(first(term.args))
            v = only(sites(term))
            e = first(filter(e -> src(e) == v || dst(e) == v, edges(s)))
            vother = src(e) == v ? dst(e) : src(e)
            set!(term_dict, e, term_dict[e] + term)
        elseif length(sites(term)) == 2 && !iszero(first(term.args))
            v1, v2 = first(sites(term)), last(sites(term))
            ed = NamedEdge(v1 => v2)
            actual_ed = ed ∈ edges(s) ? ed : reverse(ed)
            set!(term_dict, actual_ed, term_dict[actual_ed] + term)
        end
    end
    
    return term_dict
end

using NamedGraphs: NamedGraph, edges, NamedEdge, edgetype
using NamedGraphs.NamedGraphGenerators: named_grid
using NamedGraphs.GraphsExtensions: forest_cover, default_root_vertex, vertices, dfs_tree, undirected_graph,
    bfs_tree, neighbors, dst, src, rem_edges, is_tree

function BFS_prioritise_edges(g, start_region, priority_edges)
    start_vertex = first(start_region)
    Q = NamedEdge[NamedEdge(start_vertex => vn) for vn in neighbors(g, start_vertex)]
    Q = sort(Q; by = e -> dst(e) ∉ start_region)
    explored_vertices = [start_vertex]
    edges_traversed = edgetype(g)[]
    while length(vertices(g)) != length(explored_vertices)
        e = popfirst!(Q)
        v = dst(e)
        push!(explored_vertices, v)
        Q = filter(e -> dst(e) != v, Q)
        if e ∈ edges(g)
            push!(edges_traversed,  e)
        else
            push!(edges_traversed, reverse(e))
        end
        es = edgetype(g)[edgetype(g)(v => vn) for vn in neighbors(g, v)]

        for e in es
            if dst(e) ∉ explored_vertices
                if e ∈ priority_edges || reverse(e) ∈ priority_edges
                    pushfirst!(Q, e)
                else
                    push!(Q, e)
                end
            end
        end
    end
    return edges_traversed
end

function spanning_trees(g, start_region)
    edges_covered = NamedEdge[]
    priority_edges = NamedEdge[]
    ts = NamedGraph[]
    while length(edges_covered) != length(edges(g))
        t = BFS_prioritise_edges(g, start_region, priority_edges)
        append!(edges_covered, t)
        edges_covered = unique(edges_covered)
        tree = copy(g)
        tree = rem_edges(tree, setdiff(edges(g), t))
        push!(ts, tree)
        priority_edges = setdiff(edges(g), edges_covered)
    end

    return ts
end

function get_tnos(s::IndsNetwork, H::OpSum, start_region)
    ts = spanning_trees(underlying_graph(s), start_region)
    H_dict = opsum_to_edge_term_dict(s, H)
    tnos = ITensorNetwork[]
    for t in ts
        s_t = copy(s)
        s_t = rem_edges(s_t, edges(s_t))
        s_t = add_edges(s_t, edges(t))
        H_f = OpSum()
        H_dict_es = copy(keys(H_dict))
        for e in H_dict_es
            if e ∈ edges(s_t) || reverse(e) ∈ edges(s_t)
                H_f += H_dict[e]
                delete!(H_dict, e)
            end
        end
        tno = ttn(H_f, s_t)
        tno = truncate(tno; cutoff = 1e-14)
        push!(tnos, ITensorNetwork(tno))
    end

    return tnos
end