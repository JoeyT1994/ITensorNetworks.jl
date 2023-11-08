using NamedGraphs
using ITensorNetworks
using ITensors
using Random
using LinearAlgebra
using ITensorNetworks:
  contract_inner,
  neighbor_vertices,
  message_tensors,
  belief_propagation,
  symmetric_to_vidal_gauge,
  approx_network_region,
  get_environment,
  find_subgraph,
  diagblocks,
  initialize_bond_tensors,
  setindex_preserve_graph!,
  simple_update_bp,
  sqrt_and_inv_sqrt,
  simple_update_bp_full,
  expect_BP
using Dictionaries
using Observers
using NPZ
using Statistics

using SplitApplyCombine

using NamedGraphs: decorate_graph_edges

using OMEinsumContractionOrders

function exp_ITensor(A::ITensor, beta::Union{Float64,ComplexF64}; nterms=10)
  #This should be identity when combinedind(inds(A, plev = 0)) = combinedind(inds(A, plev = 1))
  # out = permute(A, [inds(A; plev=1)..., inds(A; plev=0)...])
  # out = exp(0.0 * out)
  # out = abs.(out)
  # #Need to be VERY careful. The projector to up up has a minus sign in it!
  s1, s2, s3, s4 = inds(A; plev=1)[1],
  inds(A; plev=1)[2], inds(A; plev=0)[1],
  inds(A; plev=0)[2]
  # out[s1 => 2, s2 => 2, s3 => 2, s4 => 2] = -1.0
  out = dag(op("I", s3) * op("I", s4))
  power = copy(out)
  for i in 1:nterms
    power = (1 / i) * swapprime(beta * power * prime(A), 2, 1)

    out = out + power
  end

  return out
end

function doublon_op(s::Index)
  o = op("Nup", s)
  o = o * prime(op("Ndn", s))
  swapprime!(o, 2, 1)
  return o
end

function hubbard_chain_gates(
  s::IndsNetwork; U::Float64=0.0, t::Float64=1.0, tperp::Float64=0.0, dt::Float64=0.1
)
  gates = ITensor[]
  for e in edges(s)
    vsrc, vdst = src(e), dst(e)
    @show
    vsrc_z, vdst_z = length(neighbors(g, vsrc)), length(neighbors(g, vdst))
    t = src(e)[1] != dst(e)[1] ? tperp : t
    if !iszero(t)
      hj =
        -t * op("Cdagup", s[vsrc]) * op("Cup", s[vdst]) +
        t * op("Cup", s[vsrc]) * op("Cdagup", s[vdst]) -
        t * op("Cdagdn", s[vsrc]) * op("Cdn", s[vdst]) +
        t * op("Cdn", s[vsrc]) * op("Cdagdn", s[vdst])
    end
    if !iszero(U)
      hj +=
        (U / vsrc_z) * op("Nupdn", s[vsrc]) * op("I", s[vdst]) +
        (U / vdst_z) * op("I", s[vsrc]) * op("Nupdn", s[vdst])
    end

    if !iszero(t) || !iszero(U)
      push!(gates, exp_ITensor(hj, dt * im / 2))
    end
  end
  append!(gates, reverse(gates))

  return gates
end

function hubbard_chains(g::NamedGraph; t1=1.0, t2=0.0, U=0.0)
  ℋ = OpSum()
  for e in edges(g)
    t = src(e)[1] != dst(e)[1] ? t2 : t1
    if !iszero(t)
      ℋ += t, "Cdagup", src(e), "Cup", dst(e)
      ℋ -= t, "Cup", src(e), "Cdagup", dst(e)
      ℋ += t, "Cdagdn", src(e), "Cdn", dst(e)
      ℋ -= t, "Cdn", src(e), "Cdagdn", dst(e)
    end
  end

  for v in vertices(g)
    if !iszero(U)
      ℋ += U, "Nup", v, "Ndn", v
    end
  end

  return ℋ
end

function exact_state_vector(ψ::ITensor, gate::ITensor; z=nothing)
  ψ = copy(ψ)
  if isnothing(z)
    z = (ψ * dag(ψ))[]
  end

  oψ = noprime!(ψ * gate)
  return (oψ * dag(ψ))[] / z
end

function grid_periodic_x(ny::Int64, nx::Int64)
  g = named_grid((ny, nx))
  for i in 1:ny
    NamedGraphs.add_edge!(g, (i, 1) => (i, nx))
  end
  return g
end

function main(
  g::NamedGraph,
  χparr::Int64,
  χperp::Int64,
  time_steps::Vector{Float64},
  tperp::Float64=0.0,
  U::Float64=10.0,
)
  #s = siteinds("S=1/2", g; conserve_qns=true)
  s = siteinds("Electron", g; conserve_qns=true)
  state_vector_backend = length(vertices(g)) < 15 ? true : false
  #state_vector_backend = true
  ny, nx = maximum(vertices(g))
  ITensors.disable_warn_order()

  ψ = ITensorNetwork(s, v -> (v[2]) % 2 == 0 ? "Up" : "Dn")
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])
  mts = message_tensors(
    ψψ;
    subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ)))),
    itensor_constructor=denseblocks ∘ delta,
  )
  mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"))
  gates = hubbard_chains(g; t2=tperp, U=U)
  time = 0
  Q1s = zeros((length(time_steps) + 1))
  Q2s = zeros((length(time_steps) + 1))
  Q3s = zeros((length(time_steps) + 1))
  #Q3s[1] = real(calculate_Q3(ψ, ψψ, mts))
  #Q2s[1] = real(calculate_Q2(ψ, ψψ, mts))
  Q1s[1] = mean(collect(values(real.(expect_BP("Ntot", ψ, ψψ, mts)))))
  Q1s_exact, Q2s_exact, Q3s_exact = copy(Q1s), copy(Q2s), copy(Q3s)
  times = vcat([0.0], cumsum(time_steps))

  if state_vector_backend
    ψ_sv = reduce(*, ITensor[ψ[v] for v in vertices(ψ)])
  end

  @show collect(values(real.(expect_BP("Nup", ψ, ψψ, mts))))
  for (i, dt) in enumerate(time_steps)
    #@show time
    #𝒰 = exp(-im * dt * gates; alg=Trotter{2}())
    #u⃗ = Vector{ITensor}(𝒰, s)
    u⃗ = hubbard_chain_gates(s; U, tperp, dt)
    for (j, u) in enumerate(u⃗)
      v⃗ = ITensorNetworks._gate_vertices(u, ψ)
      e = NamedEdge(v⃗[1] => v⃗[2])
      χ = src(e)[1] != dst(e)[1] ? χperp : χparr
      ψ, ψψ, mts = apply(u, ψ, ψψ, mts; maxdim=χ, cutoff=1e-12)
      if state_vector_backend
        ψ_sv = noprime(ψ_sv * u)
      end
    end
    time += dt

    mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"), niters=20)
    Q1s[i + 1] = mean(collect(values(real.(expect_BP("Ntot", ψ, ψψ, mts)))))
    #Q2s[i + 1] = real(calculate_Q2(ψ, ψψ, mts))
    #Q3s[i + 1] = real(calculate_Q3(ψ, ψψ, mts))
    #if state_vector_backend
    #  Q2s_exact[i + 1] = real(calculate_Q2_exact(ψ_sv, nx, ny, s))
    #  Q3s_exact[i + 1] = real(calculate_Q3_exact(ψ_sv, nx, ny, s))
    #end

    flush(stdout)
  end

  @show abs.(
    [exact_state_vector(ψ_sv, op("Nupdn", s[v])) for v in vertices(g)] - collect(values(real.(expect_BP("Nupdn", ψ, ψψ, mts))))
  )

  mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"), niters=30)

  ΔQ1 = abs(Q1s[length(time_steps) + 1] - Q1s[1])
  ΔQ2 = abs(Q2s[length(time_steps) + 1] - Q2s[1])
  ΔQ3 = abs(Q3s[length(time_steps) + 1] - Q3s[1])

  println("Evolution finished. Change in conserved quantitys is $ΔQ1, $ΔQ2 and $ΔQ3.")

  return Q1s, Q2s, Q3s, Q1s_exact, Q2s_exact, Q3s_exact, times
end

ITensors.enable_auto_fermion()

if length(ARGS) > 1
  nx = parse(Int64, ARGS[1])
  ny = parse(Int64, ARGS[2])
  χparr = parse(Int64, ARGS[3])
  χperp = parse(Int64, ARGS[4])
  tperp = parse(Float64, ARGS[5])
  U = parse(Float64, ARGS[6])
  save = true
else
  nx, ny = 6, 1
  tperp = 0.0
  U = 2.1
  χparr, χperp = 24, 1
end

#g = grid_periodic_x(ny, nx)
g = named_grid((ny, nx))

time_steps = [0.05 for i in 1:20]

@show χparr, χperp
flush(stdout)
Q1s, Q2s, Q3s, Q1s_exact, Q2s_exact, Q3s_exact, times = main(
  g, χparr, χperp, time_steps, tperp, U
)

save = true
if save
  file_str =
    "/mnt/home/jtindall/Documents/Data/ITensorNetworks/CoupledHubbard/ChiParr" *
    string(χparr) *
    "ChiPerp" *
    string(χperp) *
    "Nx" *
    string(nx) *
    "Ny" *
    string(ny) *
    "tperp" *
    string(round(tperp; digits=3)) *
    "U" *
    string(round(U; digits=3)) *
    "Tmax" *
    string(round(sum(time_steps); digits=3))
  file_str *= ".npz"
  npzwrite(
    file_str;
    Q1s=Q1s,
    Q2s=Q2s,
    Q3s=Q3s,
    Q1s_exact=Q1s_exact,
    Q2s_exact=Q2s_exact,
    Q3s_exact=Q3s_exact,
    times=times,
  )
end

ITensors.disable_auto_fermion()
