include("utils.jl")

Random.seed!(1634)
χ = 2
g = named_grid((4, 4))
g = rem_vertex(g, (2,2))
Lx, Ly = maximum(vertices(g))
s = siteinds("S=1/2", g)
ψ = random_tensornetwork(s; link_space = χ)
ψIψ_bpc = initialize_cache(ψ)

@show messages(ψIψ_bpc)
