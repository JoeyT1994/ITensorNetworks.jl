using NamedGraphs.GraphsExtensions: is_tree
using NamedGraphs.PartitionedGraphs: partitionvertices, partitionedges, PartitionEdge
using ITensorNetworks: ITensorNetwork, QuadraticFormNetwork, BeliefPropagationCache, update, default_message_update, maxlinkdim, indsnetwork
using ITensors: scalar

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
    inserter_kwargs = (;), optimizer = "LineSearch")

    L = length(vertices(ψ_init))
    state, ψIψ_bpc = initialize_cache(ψ_init; cache_update_kwargs = bp_update_kwargs)
    state_vertices, state_edges = collect(vertices(state)), edges(state)
    regions = new_bp_region_plan(underlying_graph(ψ_init); nsites, add_additional_traversal = false)


    init_energy = real(sum(expect(state, H; alg="bp", (cache!)=Ref(ψIψ_bpc))))
    println("Initial energy density is $(init_energy / L)")
    energies = Float64[init_energy / L]
    term_dict = opsum_to_edge_dict(indsnetwork(state), H)

    count = 0
    for i in 1:no_sweeps
        println("Beginning sweep $i")
        for region in regions
            println("Updating vertex $region")
            println("This is update $count")

            cur_local_state, ∂ψOψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts = bp_extracter(state, H, ψIψ_bpc, region; regularization = 0)

            if optimizer == "LineSearch"
                new_local_state, final_energy = bp_eigsolve_updater(cur_local_state, ∂ψOψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts)
                state, ψIψ_bpc, final_energy = optimise(H, new_local_state, cur_local_state, state,  ψIψ_bpc,
                sqrt_mts, inv_sqrt_mts, region, last(energies); bp_update_kwargs, inserter_kwargs)
            elseif optimizer == "None"
                new_local_state, final_energy = bp_eigsolve_updater(cur_local_state, ∂ψOψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts)
                state, ψIψ_bpc = bp_inserter(state, ψIψ_bpc, new_local_state, sqrt_mts, inv_sqrt_mts, region; bp_update_kwargs, inserter_kwargs...)
                final_energy = real(sum(expect(state, H; alg="bp", (cache!)=Ref(ψIψ_bpc)))) / L
            elseif optimizer == "Rotate"
                state, ψIψ_bpc, final_energy = bp_eigsolve_updater_V2(cur_local_state, ∂ψOψ_bpc_∂rs, sqrt_mts, inv_sqrt_mts, last(energies), state, ψIψ_bpc, H, region; bp_update_kwargs, inserter_kwargs)
            end

            println("Updated state, energy density of $final_energy")

            count += 1

            append!(energies, final_energy)
        end
    end

    return state, energies
end