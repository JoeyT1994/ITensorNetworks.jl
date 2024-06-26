@eval module $(gensym())
using Graphs: vertices
using ITensors: ITensors
using ITensorMPS: ITensorMPS
using ITensorNetworks: ITensorNetwork, cartesian_to_linear, dmrg, expect, siteinds, tebd
using ITensorNetworks.ITensorsExtensions: group_terms
using ITensorNetworks.ModelHamiltonians: ModelHamiltonians
using NamedGraphs.GraphsExtensions: rename_vertices
using NamedGraphs.NamedGraphGenerators: named_grid
using Test: @test, @testset, @test_broken

ITensors.disable_warn_order()

@testset "Ising TEBD" begin
  dims = (2, 3)
  n = prod(dims)
  g = named_grid(dims)

  h = 0.1

  s = siteinds("S=1/2", g)

  #
  # DMRG comparison
  #
  g_dmrg = rename_vertices(v -> cartesian_to_linear(dims)[v], g)
  ℋ_dmrg = ModelHamiltonians.ising(g_dmrg; h)
  s_dmrg = [only(s[v]) for v in vertices(s)]
  H_dmrg = ITensorMPS.MPO(ℋ_dmrg, s_dmrg)
  ψ_dmrg_init = ITensorMPS.MPS(s_dmrg, j -> "↑")
  E_dmrg, ψ_dmrg = dmrg(
    H_dmrg, ψ_dmrg_init; nsweeps=20, maxdim=[fill(10, 10); 20], cutoff=1e-8, outputlevel=0
  )

  #
  # PEPS TEBD optimization
  #
  ℋ = ModelHamiltonians.ising(g; h)
  χ = 2
  β = 2.0
  Δβ = 0.2

  ψ_init = ITensorNetwork(v -> "↑", s)
  #E0 = expect(ℋ, ψ_init)
  ψ = tebd(
    group_terms(ℋ, g),
    ψ_init;
    β,
    Δβ,
    cutoff=1e-8,
    maxdim=χ,
    ortho=false,
    print_frequency=typemax(Int),
  )
  #E1 = expect(ℋ, ψ)
  ψ = tebd(
    group_terms(ℋ, g),
    ψ_init;
    β,
    Δβ,
    cutoff=1e-8,
    maxdim=χ,
    ortho=true,
    print_frequency=typemax(Int),
  )
  #E2 = expect(ℋ, ψ)
  #@show E0, E1, E2, E_dmrg
  @test_broken (((abs((E2 - E1) / E2) < 1e-3) && (E1 < E0)) || (E2 < E1 < E0))
  @test_broken E2 ≈ E_dmrg rtol = 1e-3
end
end
