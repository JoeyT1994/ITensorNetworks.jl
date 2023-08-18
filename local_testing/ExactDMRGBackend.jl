using ITensors
using NamedGraphs
using Graphs


function HSpin(adj_mat, params, sites)
    ampo = OpSum()
    @assert size(adj_mat)[1] == size(adj_mat)[2]
    L = size(adj_mat)[1]
    for i=1:L
        for j = i+1:L
            ampo += 4*adj_mat[i,j]*params["Jx"], "Sx",i,"Sx",j
            ampo += 4*adj_mat[i,j]*params["Jy"], "Sy",i,"Sy",j
            ampo += 4*adj_mat[i,j]*params["Jz"], "Sz",i,"Sz",j
        end
    end

    for i = 1:L
        ampo += 2*params["hx"], "Sx",i
        ampo += 2*params["hy"], "Sy",i
        ampo += 2*params["hz"], "Sz",i
    end
    H = MPO(ampo, sites)
    return H
end

function HFermion(adj_mat, params, sites)
    ampo = OpSum()
    @assert size(adj_mat)[1] == size(adj_mat)[2]
    L = size(adj_mat)[1]
    for i=1:L
        for j = i+1:L
                ampo += params["t"]*adj_mat[i,j], "Cdag", i, "C", j
                ampo += params["t"]*adj_mat[i,j], "Cdag", j, "C", i
        end
    end

    #for i = 1:L
    #    ampo += params["U"], "Nup",i,"Ndn",i
    #end
    H = MPO(ampo, sites)
    return H
end

function DMRG_backend(params, g::AbstractGraph, χ::Int64; nsweeps = 20)
    L = length(vertices(g))
    sites = siteinds("S=1/2", L; conserve_qns = false)
    init_state = [isodd(i) ? "Up" : "Dn" for i = 1:L]
    ψ0 = randomMPS(sites, init_state)

    sweeps = Sweeps(nsweeps)
    setmaxdim!(sweeps,χ)

    adj_mat = adjacency_matrix(g)
    H = HSpin(adj_mat, params, sites)

    e, ψ = dmrg(H,ψ0, sweeps)

    println("DMRG Finished and found an energy of "*string(e))

end

function fermion_DMRG_backend(params, g::AbstractGraph, χ::Int64; nsweeps = 20)
    L = length(vertices(g))
    sites = siteinds("Fermion", L;conserve_nf=true, conserve_sz=true)
    init_state = [isodd(i) ? "Occ" : "Emp" for i = 1:L]
    ψ0 = randomMPS(sites, init_state; linkdims =10)

    sweeps = Sweeps(nsweeps)
    setmaxdim!(sweeps,χ)

    adj_mat = adjacency_matrix(g)
    H = HFermion(adj_mat, params, sites)

    e, ψ = dmrg(H,ψ0, sweeps)

    println("DMRG Finished and found an energy of "*string(e))

end

#params = Dict([("Jx", 1.0), ("Jy", 0.0), ("Jz", 0.0), ("hx", 0.0), ("hy", 0.0), ("hz", 3.0)])
#g = named_grid((10, 10))

#DMRG_backend(params, g, 500)