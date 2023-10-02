using ITensors
using ITensorNetworks
using Random
using SplitApplyCombine

using ITensorNetworks:
  approx_network_region,
  belief_propagation,
  sqrt_belief_propagation,
  ising_network_state,
  message_tensors

function main(; n, niters, network="ising", β=nothing, h=nothing, χ=nothing)
  g_dims = (n, n)
  @show g_dims
  g = named_grid(g_dims)
  s = siteinds("S=1/2", g)

  Random.seed!(5467)

  ψ = if network == "ising"
    ising_network_state(s, β; h)
  elseif network == "random"
    randomITensorNetwork(s; link_space=χ)
  else
    error("Network type $network not supported.")
  end

  ψψ = norm_network(ψ)

  # Site to take expectation value on
  v = (n ÷ 2, n ÷ 2)
  @show v

  #Now do Simple Belief Propagation to Measure Sz on Site v
  mts = message_tensors(
    ψψ; subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ))))
  )

  mts = @time belief_propagation(ψψ, mts; niters, contract_kwargs=(; alg="exact"))
  sz_bp = first(collect(values(expect_BP("Sz", ψ, ψψ, mts; expec_vertices=[v]))))

  println(
    "Simple Belief Propagation Gives Sz on Site " * string(v) * " as " * string(sz_bp)
  )

  mts_sqrt = message_tensors(
    ψψ; subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ))))
  )

  mts_sqrt = @time sqrt_belief_propagation(ψ, mts_sqrt; niters)
  sz_sqrt_bp = first(collect(values(expect_BP("Sz", ψ, ψψ, mts_sqrt; expec_vertices=[v]))))

  println(
    "Sqrt Belief Propagation Gives Sz on Site " * string(v) * " as " * string(sz_sqrt_bp)
  )

  return (; sz_bp, sz_sqrt_bp)
end
