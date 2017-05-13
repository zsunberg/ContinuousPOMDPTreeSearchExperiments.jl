using POMDPs
using ContinuousPOMDPTreeSearchExperiments
using POMDPToolbox
using POMCP
using POMCPOW
using ProgressMeter
using PmapProgressMeter
using ParticleFilters
using JLD
using CPUTime
using VDPTag
using Plots

N = 1000

@everywhere begin
    using VDPTag
    using POMDPs
    using ContinuousPOMDPTreeSearchExperiments
    using POMDPToolbox
    using POMCP
    using POMCPOW
    using ParticleFilters
    using CPUTime

    pomdp = VDPTagPOMDP()
    p_rng = MersenneTwister(4)
    solvers = Dict{Tuple, Union{Solver, Policy}}()
    for m in [0.01, 0.1, 1, 2, 5, 10]

        solvers = merge(solvers, Dict{Tuple, Union{Solver,Policy}}(

            ("pomcpow", m) => begin
                rollout_policy = ToNextML(mdp(pomdp))
                n = 10_000*m
                solver = POMCPOWSolver(tree_queries=n,
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
            end,

            ("bt_100", m) => begin
                n = 1000*m
                rng3 = copy(p_rng)
                rollout_policy = ToNextML(mdp(pomdp))
                node_updater = ObsAdaptiveParticleFilter(pomdp, LowVarianceResampler(100), 0.05, rng3)
                solver = POMCPDPWSolver(tree_queries=n,
                                        c=40.0,
                                        max_depth=20,
                                        k_action=8.0,
                                        alpha_action=1/20,
                                        k_observation=4.0,
                                        alpha_observation=1/20,
                                        estimate_value=FORollout(rollout_policy),
                                        node_belief_updater=node_updater,
                                        default_action=TagAction(false, 0.0),
                                        rng=rng3
                                       )
            end,

            # ("dpw", m) => begin
            #     rng3 = copy(p_rng)
            #     rollout_policy = ToNextML(mdp(pomdp))
            #     n = 10_000*m
            #     solver = POMCPDPWSolver(tree_queries=n,
            #                             c=40.0,
            #                             max_depth=20,
            #                             k_action=8.0,
            #                             alpha_action=1/20,
            #                             k_observation=4.0,
            #                             alpha_observation=1/20,
            #                             estimate_value=FORollout(rollout_policy),
            #                             default_action=TagAction(false, 0.0),
            #                             rng=rng3
            #                            )
            # end,
        ))
    end
end

solver_keys = collect(keys(solvers))
rewards = Dict{Tuple, AbstractVector{Float64}}()
# state_hists = Dict{String, AbstractVector{AbstractVector{state_type(pomdp)}}}()
times = Dict{Tuple, AbstractVector{Float64}}()
steps = Dict{Tuple, AbstractVector{Int}}()

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
    @show sum(times[sk])/sum(steps[sk])
end

for k in solver_keys
    println("$k mean: $(mean(rewards[k])) sem: $(std(rewards[k])/sqrt(N))")
    println("$k time per step: $(sum(times[k])/sum(steps[k]))")
end

filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "data", "pomdp_trends_$(Dates.format(now(), "E_d_u_HH_MM")).jld")
println("saving to $filename...")
@save(filename, solver_keys, rewards, times, steps)
println("done.")

solver_types = unique([k[1] for k in solver_keys])
series = Dict{String, Any}()
for t in solver_types
    x = []
    y = []
    for (sk, rew) in rewards
        if sk[1] == t
            push!(x, sk[2])
            push!(y, mean(rew))
        end
    end
    p = sortperm(x)
    series[t] = (x[p], y[p])
end
unicodeplots()
for (t, p) in series
    plot(p[1], p[2], ylim=(-30, 100), title=t)
    gui()
end
