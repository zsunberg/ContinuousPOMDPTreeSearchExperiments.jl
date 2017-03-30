using POMDPs
using ContinuousPOMDPTreeSearchExperiments
using POMDPToolbox
using POMCP
using POMCPOW
using LightDarkPOMDPs
using ProgressMeter
using PmapProgressMeter
using ParticleFilters
using JLD
using CPUTime
using Powseeker

N = 1

@everywhere begin
    using LightDarkPOMDPs
    using POMDPs
    using ContinuousPOMDPTreeSearchExperiments
    using POMDPToolbox
    using POMCP
    using POMCPOW
    using ParticleFilters

    pomdp = PowseekerPOMDP()
    p_rng = MersenneTwister(4)
    solvers = Dict{String, Union{Solver,Policy}}(


        "bt_100_200" => begin
            rng3 = copy(p_rng)
            action_gen = GPSFirst(rng3)
            rollout_policy = Downhill(pomdp)
            tree_queries = 200
            node_updater = ObsAdaptiveSRFilter(pomdp, LowVarianceResampler(100), 0.05, rng3)
            solver = POMCPDPWSolver(next_action=action_gen,
                                    tree_queries=tree_queries,
                                    c = exp(35.0),
                                    max_depth=mdp(pomdp).duration+2,
                                    k_action=8.0,
                                    alpha_action=1/20,
                                    k_observation=8.0,
                                    alpha_observation=1/10,
                                    estimate_value=FORollout(rollout_policy),
                                    node_sr_belief_updater=node_updater,
                                    rng=rng3
                                   )
        end,

        "pomdpdpw_10k" => begin
            rng3 = copy(p_rng)
            action_gen = GPSFirst(rng3)
            rollout_policy = Downhill(pomdp)
            tree_queries = 10_000
            solver = POMCPDPWSolver(next_action=action_gen,
                                    tree_queries=tree_queries,
                                    c = exp(35.0),
                                    max_depth=mdp(pomdp).duration+2,
                                    k_action=8.0,
                                    alpha_action=1/20,
                                    k_observation=8.0,
                                    alpha_observation=1/10,
                                    estimate_value=FORollout(rollout_policy),
                                    # node_sr_belief_updater=node_updater,
                                    rng=rng3
                                   )
        end,

        "pomcpow_10k" => begin
            rng3 = copy(p_rng)
            action_gen = GPSFirst(rng3)
            rollout_policy = SkiOver(pomdp)
            tree_queries = 10_000
            solver = POMCPOWSolver(next_action=action_gen,
                                    tree_queries=tree_queries,
                                    criterion=MaxUCB(exp(35)),
                                    final_criterion=MaxTries(),
                                    max_depth=mdp(pomdp).duration+2,
                                    k_action=8.0,
                                    alpha_action=1/20,
                                    k_observation=8.0,
                                    alpha_observation=1/10,
                                    estimate_value=FORollout(rollout_policy),
                                    rng=rng3
                                   )
        end,
    )
end

solver_keys = collect(keys(solvers))
rewards = Dict{String, AbstractVector{Float64}}()
# state_hists = Dict{String, AbstractVector{AbstractVector{state_type(pomdp)}}}()
times = Dict{String, AbstractVector{Float64}}()
steps = Dict{String, AbstractVector{Int}}()

for (j, sk) in enumerate(solver_keys)
    s_rewards = SharedArray(Float64, N) 
    # s_hists = SharedArray(AbstractVector{state_type(pomdp)}, N)
    s_times = SharedArray(Float64, N)
    s_steps = SharedArray(Float64, N)
    prog = Progress(N, desc="$sk ($j of $(length(solver_keys)))...")
    pmap(prog, 1:N) do i 
        sim_rng = MersenneTwister(i)
        up_rng = MersenneTwister(i+100_000)
        policy = solve(solvers[sk], pomdp)
        up = ObsAdaptiveParticleFilter(deepcopy(pomdp), LowVarianceResampler(10_000), 0.05, up_rng)
        sim = HistoryRecorder(max_steps=40, rng=sim_rng)
        s_times[i] = @CPUelapsed hist = simulate(sim, deepcopy(pomdp), policy, up)
        s_rewards[i] = discounted_reward(hist)
        s_steps[i] = n_steps(hist)
    end
    rewards[sk] = sdata(s_rewards)
    @show mean(rewards[sk])
    # state_hists[sk] = sdata(s_hists)
    times[sk] = sdata(s_times)
    steps[sk] = sdata(s_steps)
end

for k in solver_keys
    println("$k mean: $(mean(rewards[k])) sem: $(std(rewards[k])/sqrt(N))")
    println("$k time: $(mean(times[k]))")
end

filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "data", "compare_$(Dates.format(now(), "E_d_u_HH_MM")).jld")
println("saving to $filename...")
@save(filename, solver_keys, rewards, times, steps)
println("done.")
