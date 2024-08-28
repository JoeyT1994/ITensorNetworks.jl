include("utils.jl")

Random.seed!(1634)
ITensors.disable_warn_order()
χ = 2
g = lieb_lattice(9,9; periodic = false)
#g = named_grid((2,2))
#g = named_hexagonal_lattice_graph(8,8)
#g = named_grid_periodic_x((8,8))

Lx, Ly = maximum(vertices(g))
s = siteinds("S=1/2", g)
ψ = random_tensornetwork(ComplexF64, s; link_space = χ)
ψ = noprime(normalize(ψ; alg = "bp"))
v_measure = (5,5)

ranks = [2,4, 8]

# ψIψ_bpc = initialize_cache(ψ; rank = 2)
# pe = PartitionEdge(2=>1)
# m1 = update_message_planar(ψIψ_bpc, pe; niters = 10)
# m2 = get_column(ψIψ_bpc, PartitionVertex(2))
# m2 = [m2[(v, "ket")]*m2[(v, "operator")]*m2[(v, "bra")] for v in first.(vertices(m1))]
# @show inner(MPS(m2), MPS(m1)) / sqrt(inner(MPS(m2), MPS(m2))*inner(MPS(m1), MPS(m1)))

for r in ranks
    @show r
    ψIψ_bpc = initialize_cache(ψ; rank = r)
    pv_measure = partitionvertex(ψIψ_bpc, (v_measure, "operator"))
    seq = PartitionEdge.(post_order_dfs_edges(partitioned_graph(ψIψ_bpc), parent(pv_measure)))
    ψIψ_bpc = update_planar(ψIψ_bpc; maxiter = 1, tol = 1e-8, niters=10, overlap_tol = 1e-10)
    @show expect_planar(ψIψ_bpc, s, "Z", v_measure)
    @show expect_planar_exact(ψIψ_bpc, s, "Z", v_measure)
end


#ψIψ_bpc = update_planar(ψIψ_bpc; maxiter = 1, tol = 1e-16, niters=50, overlap_tol = 1e-16)

