using NamedGraphs
using ITensorNetworks
using ITensors
using Random
using LinearAlgebra
using Dictionaries
using Observers
using NPZ
using Statistics

using SplitApplyCombine

using NamedGraphs: decorate_graph_edges

using OMEinsumContractionOrders

using ITensorTDVP

function general_expect(ψ::MPS, Os::Vector{String}, sites::Vector{Int64}, s)
  ψ = copy(ψ)
  orthogonalize!(ψ, first(sites))
  O = 1.0
  for i in 1:length(Os)
    O *= op(Os[i], s[sites[i]])
  end
  
  ψO = apply(O, ψ)

  return inner(ψ, ψO)
end

function calculate_Q2(nx::Int64, ny::Int64, 
  ψ::MPS, reverse_vertex_map, s; central_row=true
)
  ny_iter = central_row ? [ceil(Int64, ny * 0.5)] : [i for i in 1:ny]
  nx_iter = [i for i in 1:nx]

  strings = [["X", "X"], ["Y", "Y"], ["Z", "Z"]]
  out = 0
  vertex_sets = [
    [(j, mod(i, nx) + 1), (j, mod(i + 1, nx) + 1)] for i in nx_iter for j in ny_iter
  ]

  for v_set in vertex_sets
    site_set = [reverse_vertex_map[v] for v in v_set]
    for string in strings
      out +=general_expect(ψ, string, site_set, s)
    end
  end

  out /= inner(ψ, ψ)

  return out / (length(nx_iter) * length(ny_iter))
end

function calculate_Q3(nx::Int64, ny::Int64, 
  ψ::MPS, reverse_vertex_map, s; central_row=true
)
  ny_iter = central_row ? [ceil(Int64, ny * 0.5)] : [i for i in 1:ny]
  nx_iter = [i for i in 1:nx]

  pos_strings = [["X", "Y", "Z"], ["Y", "Z", "X"], ["Z", "X", "Y"]]
  neg_strings = [["X", "Z", "Y"], ["Y", "X", "Z"], ["Z", "Y", "X"]]
  out = 0
  vertex_sets = [[(j, mod(i, nx) + 1), (j, mod(i + 1, nx) + 1), (j, mod(i + 2, nx) + 1)] for i in nx_iter for j in ny_iter]

  for v_set in vertex_sets
    site_set = [reverse_vertex_map[v] for v in v_set]
    for string in pos_strings
      out += general_expect(ψ, string, site_set, s)
    end

    for string in neg_strings
      out += -1.0*general_expect(ψ, string, site_set, s)
    end
  end

  out /= inner(ψ, ψ)

  return out / (length(nx_iter) * length(ny_iter))
end

function HSpin(adj_mat, sites)
  ampo = OpSum()
  @assert size(adj_mat)[1] == size(adj_mat)[2]
  L = size(adj_mat)[1]
  for i in 1:L
    for j in (i + 1):L
      if !iszero(adj_mat[i, j])
        ampo += 0.5*adj_mat[i, j], "S+", i, "S-", j
        ampo += 0.5*adj_mat[i, j], "S-", i, "S+", j
        ampo += adj_mat[i, j], "Sz", i, "Sz", j
      end
    end
  end

  H = MPO(ampo, sites)
  return H
end


function get_adj_mat(g::NamedGraph, vertex_map::Dict; Jperp::Float64 = 0.0)
  L = length(vertices(g))
  es = edges(g)
  out = zeros(Float64, (L, L))
  for i in 1:L
    for j in 1:L
      vsrc, vdst = vertex_map[i], vertex_map[j]
      if NamedEdge(vsrc => vdst) ∈ es || NamedEdge(vdst => vsrc) ∈ es
        if vsrc[1] == vdst[1]
          out[i, j], out[j, i] = 1.0, 1.0
        else
          out[i, j], out[j, i] = Jperp, Jperp
        end
      end
    end
  end

  return out
end

function grid_periodic_x(ny::Int64, nx::Int64)
  g = named_grid((ny, nx))
  for i in 1:ny
    NamedGraphs.add_edge!(g, (i, 1) => (i, nx))
  end
  return g
end

function main(
  nx::Int64, ny::Int64, χ::Int64, time_steps::Vector{Float64}, Jperp::Float64=0.0
)
  snake_up = false
  g = grid_periodic_x(ny, nx)
  s = siteinds("S=1/2", nx*ny)
  vertex_map = snake_up ? Dict(zip([i for i in 1:nx*ny], [(i, j) for j in 1:nx for i in 1:ny])) : Dict(zip([i for i in 1:nx*ny], [(i, j) for i in 1:ny for j in 1:nx]))
  reverse_vertex_map = snake_up ? Dict(zip([(i, j) for j in 1:nx for i in 1:ny], [i for i in 1:nx*ny])) : Dict(zip([(i, j) for i in 1:ny for j in 1:nx], [i for i in 1:nx*ny]))
  adj_mat = get_adj_mat(g, vertex_map; Jperp)
  ψ_t = MPS(ComplexF64, s, [(vertex_map[i][2] + 2) % 3 == 0 ? "X+" : (vertex_map[i][2] + 2) % 3 == 1 ? "Y+" : "Z+" for i in 1:nx*ny])
  H = HSpin(adj_mat, s)
  
  time = 0
  Q1s = zeros((length(time_steps) + 1))
  Q2s = zeros((length(time_steps) + 1))
  Q3s = zeros((length(time_steps) + 1))
  Q1s[1] = mean(ITensors.expect(ψ_t, "Z"))
  Q2s[1] = real(calculate_Q2(nx, ny, ψ_t, reverse_vertex_map, s))
  Q3s[1] = real(calculate_Q3(nx, ny, ψ_t, reverse_vertex_map, s))
  times = vcat([0.0], cumsum(time_steps))

  for (i, dt) in enumerate(time_steps)
    @show time
    ψ_t = ITensorTDVP.tdvp(H,-im * dt,ψ_t;time_step=-im * dt,normalize=true,maxdim=χ, cutoff = 1e-16, outputlevel=1)
    Q1s[i + 1] = mean(ITensors.expect(ψ_t, "Z"))
    Q2s[i + 1] = real(calculate_Q2(nx, ny, ψ_t, reverse_vertex_map, s))
    Q3s[i + 1] = real(calculate_Q3(nx, ny, ψ_t, reverse_vertex_map, s))

    time += dt
    flush(stdout)
  end

  @show Q1s[length(time_steps) + 1], Q2s[length(time_steps) + 1], Q3s[length(time_steps) + 1]
  ΔQ1 = abs(Q1s[length(time_steps) + 1] - Q1s[1])
  ΔQ2 = abs(Q2s[length(time_steps) + 1] - Q2s[1])
  ΔQ3 = abs(Q3s[length(time_steps) + 1] - Q3s[1])

  println("Evolution finished. Change in conserved quantities is $ΔQ1, $ΔQ2 and $ΔQ3.")

  return Q1s, Q2s, Q3s, times
end

if length(ARGS) > 1
  nx = parse(Int64, ARGS[1])
  ny = parse(Int64, ARGS[2])
  χ = parse(Int64, ARGS[3])
  Jperp = parse(Float64, ARGS[4])
  nsteps = 1000
  save = true
else
  nx, ny = 36, 2
  Jperp = 0.0
  χ = 64
  save = false
  nsteps = 100
end

time_steps = [0.01 for i in 1:nsteps]

@show χ, Jperp
flush(stdout)
Q1s, Q2s, Q3s, times = main(
  nx, ny, χ, time_steps, Jperp
)

if save
  file_str =
    "/mnt/home/jtindall/Documents/Data/ITensorNetworks/CoupledHeisenberg/TDVPChi" *
    string(χ) *
    "Nx" *
    string(nx) *
    "Ny" *
    string(ny) *
    "JPerp" *
    string(round(Jperp; digits=3)) *
    "Tmax" *
    string(round(sum(time_steps); digits=3))
  file_str *= ".npz"
  npzwrite(
    file_str;
    Q1s=Q1s,
    Q2s=Q2s,
    Q3s=Q3s,
    times=times,
  )
end
