using LaserTag
using Plots
using POMDPToolbox
using Reel
using ProgressMeter
using PmapProgressMeter
using ParticleFilters
using ContinuousPOMDPTreeSearchExperiments
using Plots


@everywhere begin
    using POMDPs
    using POMDPToolbox
    using ContinuousPOMDPTreeSearchExperiments
    using ParticleFilters
    using POMCPOW
    using POMCP
    using LaserTag

    N = 100

    solvers = Dict{String, Union{Policy, Solver}}(

        "pomcpow" => begin
            ro = MoveTowards()
            solver = POMCPOWSolver(tree_queries=500_000,
                                   criterion=MaxUCB(10.0),
                                   final_criterion=MaxTries(),
                                   max_depth=100,
                                   enable_action_pw=false,
                                   k_observation=4.0,
                                   alpha_observation=1/20,
                                   estimate_value=FORollout(ro),
                                   check_repeat_act=false,
                                   check_repeat_obs=false,
                                   rng=MersenneTwister(13)
                                  )
            solver
        end,

        "move_towards_sampled" => MoveTowardsSampled(),

        "qmdp" => QMDPSolver()

        #=
        "discrete_pomcp" => begin
            ro = translate_policy(ToNextML(mdp(pomdp)), mdp(pomdp), dpomdp, dpomdp)
            solver = POMCPSolver(tree_queries=10_000,
                                   c=40.0,
                                   max_depth=20,
                                   estimate_value=FORollout(ro),
                                   rng=MersenneTwister(13)
                                  )
            dpolicy = solve(solver, dpomdp)
            translate_policy(dpolicy, dpomdp, pomdp, dpomdp)
        end,
        =#
    )
end

@show N

for (k,sol) in solvers
    prog = Progress(N, desc="Simulating...")
    rewards = pmap(prog, 1:N) do i
        pomdp = gen_lasertag(rng=MersenneTwister(i+600_000))
        if isa(sol,Solver)
            p = solve(sol, pomdp)
        else
            p = sol
        end
        hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(i))
        # up_rng = MersenneTwister(i+100_000)
        # up = ObsAdaptiveParticleFilter(deepcopy(pomdp), LowVarianceResampler(100_000), 0.05, up_rng)
        up = DiscreteUpdater(pomdp)
        hist = simulate(hr, pomdp, p, up)
        discounted_reward(hist)
    end
    @show k 
    @show mean(rewards)
end
