include("utils.jl")

function heavy_hex_lattice_graph(n::Int64, m::Int64)
    g = named_hexagonal_lattice_graph(n, m)
    g = decorate_graph_edges(g)
    return g
  end
  
  function ibm_processor_graph(n::Int64, m::Int64)
    g = heavy_hex_lattice_graph(n, m)
    dims = maximum(vertices(g))
    v1, v2 = (1, dims[2]), (dims[1], 1)
    add_vertices!(g, [v1, v2])
    add_edge!(g, v1 => v1 .- (0, 1))
    add_edge!(g, v2 => v2 .+ (0, 1))
  
    return g
  end
  
  eagle_processor_graph() = ibm_processor_graph(3, 6)
  hummingbird_processor_graph() = ibm_processor_graph(2, 4)
  osprey_processor_graph() = ibm_processor_graph(6, 12)

function main()
    #g = ibm_processor_grid()
    g = ibm_processor_grid_periodic_x()
    s = siteinds("S=1/2", g)

    ψ = ITensorNetwork(v -> "↑", s)
    maxdim, cutoff = 32, 1e-10
    apply_kwargs = (; maxdim, cutoff, normalize = true)
    #Parameters for BP, as the graph is not a tree (it has loops), we need to specify these
    bp_update_kwargs = (; maxiter = 10, tol = 1e-5)
    ψIψ = build_bp_cache(ψ; bp_update_kwargs...)

    v_measure = first(center(g))

    θh = 0.6
    HX, HZZ = ising(g; h= 2 * (θh / 2), J1=0), ising(g; h=0, J1=- 4 * (pi / 4))
    RX, RZZ = exp(-im * HX; alg=Trotter{1}()), exp(-im * HZZ; alg=Trotter{1}())
    RX_gates, RZZ_gates = Vector{ITensor}(RX, s), Vector{ITensor}(RZZ, s)

    no_trotter_steps = 20
    for i in 1:no_trotter_steps
        println("On Trotter Step $i")
        gates = i != no_trotter_steps ? vcat(RX_gates, RZZ_gates) : RX_gates
        for gate in gates
            ψ, ψIψ = apply(gate, ψ, ψIψ; reset_all_messages = false, apply_kwargs...)
        end
        ψIψ = update(ψIψ; bp_update_kwargs...)
    end

    sz_bp = real(only(expect(ψ, "Z", [v_measure]; cache! = Ref(ψIψ))))
    println("BP computed value of <Z62> is $sz_bp")
    group_by_xpos = true
    ranks = [1,2,3,4,8,16]
    szs_boundary_mps = []
    for r in ranks
        println("Rank is $r")
        ψIψ_bpc = initialize_cache(ψ; rank = r, group_by_xpos)
        seq = post_order_dfs_edges(partitioned_graph(ψIψ_bpc), parent(only(partitionvertices(ψIψ_bpc, [(v_measure, "ket")]))))
        #ψIψ_bpc = update_planar(ψIψ_bpc, PartitionEdge.(seq); niters=25, overlap_tol = 1e-16, group_by_xpos)
        ψIψ_bpc = update_planar(ψIψ_bpc; maxiter = 5, tol = 1e-8, niters=25, overlap_tol = 1e-16, group_by_xpos)
        lazy_rdm = one_site_rdm_planar(ψIψ_bpc, v_measure; group_by_xpos)
        sz_boundary_mps = real((lazy_rdm * ITensors.op("Z", s[v_measure]))[])
        push!(szs_boundary_mps, sz_boundary_mps)
        println("Rank $r Boundary MPS computed value of <Z62> is $sz_boundary_mps")
    end
end

main()