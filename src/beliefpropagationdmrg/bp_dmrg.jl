using NamedGraphs.GraphsExtensions: is_tree
using NamedGraphs.PartitionedGraphs: partitionvertices, partitionedges, PartitionEdge
using ITensorNetworks: ITensorNetwork, QuadraticFormNetwork, BeliefPropagationCache, update, default_message_update, maxlinkdim, indsnetwork
using ITensors: scalar
using LinearAlgebra

include("bp_extracter.jl")
include("bp_inserter.jl")
include("bp_updater.jl")
include("graphsextensions.jl")
include("utils.jl")

default_bp_update_kwargs(ψ::ITensorNetwork) = is_tree(ψ) ? (;) : (; maxiter = 25, tol = 1e-8)

function initialize_cache(ψ_init::ITensorNetwork; cache_update_kwargs = default_bp_update_kwargs(ψ_init))
    ψ = copy(ψ_init)
    ψIψ = QuadraticFormNetwork(ψ)
    ψIψ_bpc = BeliefPropagationCache(ψIψ)
    ψ, ψIψ_bpc = renormalize_update_norm_cache(ψ, ψIψ_bpc; cache_update_kwargs)

    return ψ, ψIψ_bpc
end

function bp_dmrg(ψ_init::ITensorNetwork, H::OpSum; nsites = 1, no_sweeps = 1, bp_update_kwargs = default_bp_update_kwargs(ψ_init),
    inserter_kwargs = (;))

    L = length(vertices(ψ_init))
    state, ψIψ_bpc = initialize_cache(ψ_init; cache_update_kwargs = bp_update_kwargs)
    state_vertices, state_edges = collect(vertices(state)), edges(state)
    regions = new_bp_region_plan(underlying_graph(ψ_init); nsites, add_additional_traversal = false)


    init_energy = real(sum(expect(state, H; alg="bp", (cache!)=Ref(ψIψ_bpc))))
    println("Initial BP energy density is $(init_energy / L)")
    energies = Float64[init_energy / L]
    term_dict = opsum_to_edge_dict(indsnetwork(state), H)

    for i in 1:no_sweeps
        println("Beginning sweep $i")
        for region in regions
            println("Updating vertex $region")

            cur_local_state, ∂ψOψ_bpc_∂rs, ∂ψIψ_bpc_∂r = bp_extracter(state, H, ψIψ_bpc, region)

            new_local_state, final_energy = bp_eigsolve_updater(cur_local_state, ∂ψOψ_bpc_∂rs, ∂ψIψ_bpc_∂r)
            state, ψIψ_bpc = bp_inserter(state, ψIψ_bpc, new_local_state, region; bp_update_kwargs, inserter_kwargs...)

            println("Updated state, energy is $final_energy")

            append!(energies, final_energy)
        end
    end

    return state, energies
end