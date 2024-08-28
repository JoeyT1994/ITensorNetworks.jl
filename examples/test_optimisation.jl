include("utils.jl")

Random.seed!(1634)
ITensors.disable_warn_order()
χ = 5
#g = lieb_lattice(11,11; periodic = false)
g = heavy_hexagonal_lattice_graph(3,6)
#g = named_grid((10, 5))
#g = named_hexagonal_lattice_graph(8,8)
#g = named_grid_periodic_x((8,8))

Lx, Ly = maximum(vertices(g))
@show Lx, Ly
group_by_xpos = false
s = siteinds("S=1/2", g)
ψ = random_tensornetwork(ComplexF64, s; link_space = χ)
ψ = noprime(normalize(ψ; alg = "bp"))
v_measure = first(center(g))

expect_bp = only(expect(ψ, "Y", [v_measure]; alg = "bp"))
println("BP expectation value is $expect_bp")

ranks = [1,2,3,4,5,6, 8, 10, 12, 14, 16]
rdms = ITensor[]

for (i, r) in enumerate(ranks)
    @show r
    ψIψ_bpc = initialize_cache(ψ; rank = r, group_by_xpos)
    ψIψ_bpc = update_planar(ψIψ_bpc; maxiter = 1, tol = 1e-8, niters=50, overlap_tol = 1e-10, group_by_xpos)
    rdm = one_site_rdm_planar(ψIψ_bpc, v_measure; group_by_xpos)
    @show expect_planar(ψIψ_bpc,s,  "Y", v_measure; group_by_xpos)
    push!(rdms, rdm)
    if i > 1
        @show norm(rdms[i] - rdms[i-1])
    end
end
