include("utils.jl")

Random.seed!(1634)
χ = 2
g = named_grid((4, 4))
#g = rem_vertex(g, (2,2))

Lx, Ly = maximum(vertices(g))
s = siteinds("S=1/2", g)
ψ = random_tensornetwork(s; link_space = χ)
ψIψ_bpc = initialize_cache(ψ; rank =4)


pe = PartitionEdge(2=>3)
pe_in = only(setdiff(boundary_partitionedges(ψIψ_bpc, src(pe); dir=:in), [reverse(pe)]))
Φ, A, Ψ = message(ψIψ_bpc, pe_in), get_column(ψIψ_bpc, src(pe)), message(ψIψ_bpc, pe)

optimise(Φ, A, Ψ)