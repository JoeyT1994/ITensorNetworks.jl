include("utils.jl")

Random.seed!(1634)
ITensors.disable_warn_order()
χ = 25
#g = lieb_lattice(10,9; periodic = true)
g = named_hexagonal_lattice_graph(8,8)

Lx, Ly = maximum(vertices(g))
s = siteinds("S=1/2", g)
ψ = random_tensornetwork(s; link_space = χ)
ψ = noprime(normalize(ψ; alg = "bp"))
v_measure = (5,5)

ranks = [2,3,4,5,6,7,8,9,10]

for r in ranks
    @show r
    ψIψ_bpc = initialize_cache(ψ; rank = r)
    #pv_measure = partitionvertex(ψIψ_bpc, (v_measure, "operator"))
    #seq = PartitionEdge.(post_order_dfs_edges(partitioned_graph(ψIψ_bpc), parent(pv_measure)))
    #ψIψ_bpc = update_planar(ψIψ_bpc, seq; niters=25, overlap_tol = 1e-16)
    ψIψ_bpc = update_planar(ψIψ_bpc; maxiter = 5, tol = 1e-8, niters=25, overlap_tol = 1e-10)
    @show expect_planar(ψIψ_bpc, s, "Z", v_measure)
end


#ψIψ_bpc = update_planar(ψIψ_bpc; maxiter = 1, tol = 1e-16, niters=50, overlap_tol = 1e-16)

