@eval module $(gensym())
using NamedGraphs.NamedGraphGenerators: named_grid
using ITensorNetworks.GraphsExtensions: next_nearest_neighbors, nth_nearest_neighbors
using Test: @test, @testset

@testset "GraphsExtensions" begin
  @testset "Test nth nearest neighbours"  begin
    L = 10
    g = named_grid((L,1))
    vstart= (1,1)
    @test only(nth_nearest_neighbors(g, vstart, L-1)) == (L,1)
    @test only(next_nearest_neighbors(g, vstart)) == (3,1)

    g = named_grid((L, L))
    v_middle = (L / 2, L /2)
    @test length(next_nearest_neighbors(g, v_middle)) == 8
    @test length(nth_nearest_neighbors(g, v_middle, L)) == 4
  end
end
end
