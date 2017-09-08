using POMDPs
using ContinuousPOMDPTreeSearchExperiments
using POMDPToolbox
using POMCPOW
using VDPTag
using MCTS
using DataFrames
using ParticleFilters
using CPUTime

N = 4

function create_pft(m)
        rng = MersenneTwister(13)
        rollout_policy = ToNextML(mdp(pomdp))
        node_updater = ObsAdaptiveParticleFilter(pomdp, LowVarianceResampler(m), 0.05, rng)
        solver = DPWSolver(n_iterations=typemax(Int),
                           exploration_constant=40.0,
                           depth=20,
                           k_action=8.0,
                           alpha_action=1/20,
                           k_state=4.0,
                           alpha_state=1/20,
                           check_repeat_state=false,
                           check_repeat_action=false,
                           estimate_value=FORollout(rollout_policy),
                           rng=rng
                          )
        belief_mdp = GenerativeBeliefMDP(deepcopy(pomdp), node_updater)
        return solve(solver, belief_mdp)
end

function pft_stats_collector(planner)
    d = Dict(:cpu_us=>Int[],
             :n_nodes=>Int[],
            )
    return PolicyWrapper(planner, payload=d) do p, d, b
        start_us = CPUtime_us()
        a = action(p, b)
        cpu_us = CPUtime_us() - start_us
        push!(d[:cpu_us], cpu_us)
        tree = get(p.tree)
        push!(d[:n_nodes], length(tree.total_n))
        return a
    end
end

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
                               default_action=TagAction(false,0.0),
                               rng=MersenneTwister(13)
                              )
        solve(solver, deepcopy(pomdp))
    end,

    "pft_10" => pft_stats_collector(create_pft(10)),
    "pft_100" => pft_stats_collector(create_pft(100)),
    "pft_1000" => pft_stats_collector(create_pft(1000))
)

alldata = DataFrame()
# for t in logspace(-2,1,7)
for t in [0.01] 
    for (k, planner) in planners
        println("$k ($t)")
        if isa(planner, PolicyWrapper)
            planner.policy.solver.max_time = t
        else
            planner.solver.max_time = t
        end
        sims = []
        for i in 1:N
            srand(planner, i+50_000)
            sim = Sim(deepcopy(pomdp),
                      deepcopy(planner),
                      ObsAdaptiveParticleFilter(deepcopy(pomdp),
                                                LowVarianceResampler(10_000),
                                                0.05, MersenneTwister(i+90_000)),
                      rng=MersenneTwister(i+70_000),
                      max_steps=100,
                      metadata=Dict(:solver=>k,
                                    :time=>t,
                                    :i=>i)
                     )
            push!(sims, sim)
        end

        data = run_parallel(sims) do sim, h
            stuff = sim.metadata
            if isa(sim.policy, PolicyWrapper)
                p = sim.policy
                stuff[:mean_cpu_us] = mean(p.payload[:cpu_us])
                stuff[:max_cpu_us] = maximum(p.payload[:cpu_us])
                stuff[:min_cpu_us] = minimum(p.payload[:cpu_us])
                stuff[:mean_nodes] = mean(p.payload[:n_nodes])
                stuff[:max_nodes] = maximum(p.payload[:n_nodes])
                stuff[:min_nodes] = minimum(p.payload[:n_nodes])
            end
            stuff[:reward] = discounted_reward(h)
            return stuff
        end

        @show data

        rs = data[:reward]
        println(@sprintf("reward: %6.3f ± %6.3f", mean(rs), std(rs)/sqrt(length(rs))))
        alldata = vcat(alldata, data)
    end
end

filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "data", "vdp_trends_$(Dates.format(now(), "E_d_u_HH_MM")).csv")
println("saving to $filename...")
writetable(filename, alldata)
println("done.")

# filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "data", "compare_$(Dates.format(now(), "E_d_u_HH_MM")).jld")
# @save(filename, solver_keys, rewards, times, steps)
# println("done.")
