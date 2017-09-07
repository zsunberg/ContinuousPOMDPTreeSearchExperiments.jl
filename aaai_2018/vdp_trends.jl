using POMDPs
using ContinuousPOMDPTreeSearchExperiments
using POMDPToolbox
using POMCPOW
using JLD
using VDPTag
using MCTS
using DataFrames
using ParticleFilters

N = 4

pomdp = VDPTagPOMDP()
planners = Dict{String, Union{Solver,Policy}}(

    "pomcpow" => begin
        rollout_policy = ToNextML(mdp(pomdp))
        solver = POMCPOWSolver(tree_queries=10_000_000,
                               criterion=MaxUCB(40.0),
                               final_criterion=MaxTries(),
                               max_depth=20,
                               k_action=8.0,
                               alpha_action=1/20,
                               k_observation=4.0,
                               alpha_observation=1/20,
                               estimate_value=FORollout(rollout_policy),
                               check_repeat_act=false,
                               check_repeat_obs=false,
                               rng=MersenneTwister(13)
                              )
        solve(solver, deepcopy(pomdp))
    end,

    "bt_100" => begin
        rng = MersenneTwister(13)
        rollout_policy = ToNextML(mdp(pomdp))
        node_updater = ObsAdaptiveParticleFilter(pomdp, LowVarianceResampler(100), 0.05, rng)
        solver = DPWSolver(n_iterations=10_000_000,
                           exploration_constant=40.0,
                           depth=20,
                           k_action=8.0,
                           alpha_action=1/20,
                           k_state=4.0,
                           alpha_state=1/20,
                           estimate_value=FORollout(rollout_policy),
                           rng=rng
                          )
        belief_mdp = GenerativeBeliefMDP(deepcopy(pomdp), node_updater)
        solve(solver, belief_mdp)
    end
)

# for t in logspace(-2,0,5)
alldata = DataFrame()
for t in [0.1, 1.0]
    for (k, planner) in planners
        println("$k ($t)")
        planner.solver.max_time = t
        sims = []
        for i in 1:N
            srand(planner, i+50_000)
            push!(sims, Sim(pomdp,
                            deepcopy(planner),
                            ObsAdaptiveParticleFilter(deepcopy(pomdp), LowVarianceResampler(10_000), 0.05, MersenneTwister(i+90_000)),
                            rng=MersenneTwister(i+70_000),
                            max_steps=100,
                            metadata=Dict(:solver=>k,
                                          :time=>t,
                                          :i=>i)
                           ))
        end
        data = run_parallel(sims)
        @show data
        rs = data[:reward]
        println(@sprintf("reward: %6.3f ± %6.3f", mean(rs), std(rs)/sqrt(length(rs))))
        alldata = vcat(alldata, data)
    end
end

filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "data", "vdp_trends_$(Dates.format(now(), "E_d_u_HH_MM")).csv")


# filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "data", "compare_$(Dates.format(now(), "E_d_u_HH_MM")).jld")
# println("saving to $filename...")
# @save(filename, solver_keys, rewards, times, steps)
# println("done.")
