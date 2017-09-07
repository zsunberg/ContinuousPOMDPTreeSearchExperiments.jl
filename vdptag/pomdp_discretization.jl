using VDPTag
using Plots
using POMDPToolbox
using Reel
using ProgressMeter
using PmapProgressMeter
using ParticleFilters
using ContinuousPOMDPTreeSearchExperiments
using Plots

@show N = 2

@everywhere begin
    using POMDPs
    using POMDPToolbox
    using ContinuousPOMDPTreeSearchExperiments
    using ParticleFilters
    using POMCPOW
    using BasicPOMCP
    using VDPTag

    pomdp = VDPTagPOMDP()

    n = 10000
    seed = 42

    planners = Dict{String, Policy}(

        "pomcpow" => begin
            ro = ToNextML(mdp(pomdp))
            solver = POMCPOWSolver(tree_queries=n,
                                   criterion=MaxUCB(40.0),
                                   final_criterion=MaxTries(),
                                   max_depth=20,
                                   k_action=8.0,
                                   alpha_action=1/20,
                                   k_observation=4.0,
                                   alpha_observation=1/20,
                                   estimate_value=FORollout(ro),
                                   check_repeat_act=false,
                                   check_repeat_obs=false,
                                   rng=MersenneTwister(13)
                                  )
            solve(solver, pomdp)
        end,

        # "discrete_pomcpow" => begin
        #     dpomdp = AODiscreteVDPTagPOMDP(n_angles=1000, n_obs_angles=1000)
        #     ro = translate_policy(ToNextML(mdp(pomdp)), mdp(pomdp), dpomdp, dpomdp)
        #     solver = POMCPOWSolver(tree_queries=10_000,
        #                            criterion=MaxUCB(40.0),
        #                            final_criterion=MaxTries(),
        #                            max_depth=20,
        #                            k_action=8.0,
        #                            alpha_action=1/20,
        #                            k_observation=4.0,
        #                            alpha_observation=1/20,
        #                            estimate_value=FORollout(ro),
        #                            check_repeat_act=false,
        #                            check_repeat_obs=false,
        #                            rng=MersenneTwister(13)
        #                           )
        #     translate_policy(solve(solver, dpomdp), dpomdp, pomdp, dpomdp)
        # end,


        "discrete_pomcp" => begin
            dpomdp = AODiscreteVDPTagPOMDP()
            ro = translate_policy(ToNextML(mdp(pomdp)), mdp(pomdp), dpomdp, dpomdp)
            solver = POMCPSolver(tree_queries=n,
                                   c=40.0,
                                   max_depth=20,
                                   estimate_value=FORollout(ro),
                                   rng=MersenneTwister(13)
                                  )
            dpolicy = solve(solver, dpomdp)
            translate_policy(dpolicy, dpomdp, pomdp, dpomdp)
        end,
    )
end

for (k,p) in planners
    prog = Progress(N, desc="Simulating...")
    # rewards = pmap(prog, 1:N) do i
    rewards = map(1:N) do i
        hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(i))
        up_rng = MersenneTwister(i+100_000)
        up = ObsAdaptiveParticleFilter(deepcopy(pomdp), LowVarianceResampler(10_000), 0.05, up_rng)
        hist = simulate(hr, pomdp, p, up)
        discounted_reward(hist)
    end
    @show k 
    @show mean(rewards)
end
