using POMDPs
using ContinuousPOMDPTreeSearchExperiments
using POMDPToolbox
using BasicPOMCP
using POMCPOW
using ParticleFilters
using CPUTime
using VDPTag
using DataFrames

N = 4

pomdp = VDPTagPOMDP()

function extract_n_nodes(p::POMCPOWPlanner, d, b)
    a = action(p, b)
    tree = get(p.tree)
    push!(d[:n_nodes], length(tree.total_n))
    return a
end

function extract_n_nodes(p::POMCPPlanner, d, b)
    tree = BasicPOMCP.POMCPTree(p.problem, p.solver.tree_queries)
    local a::action_type(p.problem)
    try
        a = BasicPOMCP.search(p, b, tree)
    catch ex
        a = convert(action_type(p.problem), BasicPOMCP.default_action(p.solver.default_action, b, ex))
    end
    push!(d[:n_nodes], length(tree.total_n))
    return a
end


solvers = Dict{String, Union{Solver,Policy}}(

    "pomcpow" => begin
        rng = MersenneTwister(13)
        ro = ToNextMLSolver(rng)
        solver = POMCPOWSolver(tree_queries=10_000_000,
                               criterion=MaxUCB(40.0),
                               final_criterion=MaxTries(),
                               max_depth=10,
                               max_time=0.1,
                               k_action=8.0,
                               alpha_action=1/20,
                               k_observation=4.0,
                               alpha_observation=1/20,
                               estimate_value=FORollout(ro),
                               check_repeat_act=true,
                               check_repeat_obs=true,
                               next_action=RootToNextMLFirst(rng),
                               default_action=1,
                               rng=rng
                              )
    end,

    "pomcp" => begin
        rng = MersenneTwister(13)
        ro = ToNextMLSolver(rng)
        POMCPSolver(max_depth=10,
                    max_time=0.1,
                    c=40.0,
                    tree_queries=typemax(Int),
                    default_action=1,
                    estimate_value=FORollout(ro),
                    rng=rng
                   )
    end
)

alldata = DataFrame()
# for n_angles_float in logspace(0.5, 3, 6)
for n_angles_float in [10]
    n_angles = round(Int, n_angles_float)
    for (k, solver) in solvers
        println("$k ($n_angles)")
        dpomdp = AODiscreteVDPTagPOMDP(n_angles=n_angles, n_obs_angles=n_angles)

        planner = solve(solver, dpomdp)

        wrapper = PolicyWrapper(extract_n_nodes, planner, payload=Dict(:n_nodes=>Int[]))

        sims = []
        for i in 1:N
            srand(wrapper, i+50_000)
            cplanner = translate_policy(wrapper, dpomdp, pomdp, dpomdp)

            filter = ObsAdaptiveParticleFilter(deepcopy(pomdp),
                                               LowVarianceResampler(10_000),
                                               0.05, MersenneTwister(i+90_000))            

            sim = Sim(deepcopy(pomdp),
                      deepcopy(cplanner),
                      filter,
                      rng=MersenneTwister(i+70_000),
                      max_steps=100,
                      metadata=Dict(:solver=>k,
                                    :n_angles=>n_angles,
                                    :i=>i)
                     )

            push!(sims, sim)
        end

        data = run_parallel(sims) do sim, h
            stuff = sim.metadata
            wrapper = sim.policy.policy
            stuff[:mean_nodes] = mean(wrapper.payload[:n_nodes])
            stuff[:reward] = discounted_reward(h)
            return stuff
        end

        @show data

        rs = data[:reward]
        println(@sprintf("reward: %6.3f ± %6.3f", mean(rs), std(rs)/sqrt(length(rs))))
        alldata = vcat(alldata, data)
    end
end

filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "data", "vdp_discretization_$(Dates.format(now(), "E_d_u_HH_MM")).csv")
println("saving to $filename...")
writetable(filename, alldata)
println("done.")
