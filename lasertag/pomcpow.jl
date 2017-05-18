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

    N = 2
    n_rows = 7
    n_cols = 11
    prng = MersenneTwister(14)
    robs = [Coord(rand(prng, 1:n_cols), rand(prng, 1:n_rows)) for i in 1:N]

    solvers = Dict{String, Solver}(

        "pomcpow" => begin
            ro = MoveTowards()
            solver = POMCPOWSolver(tree_queries=10_000,
                                   criterion=MaxUCB(40.0),
                                   final_criterion=MaxTries(),
                                   max_depth=20,
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
        pomdp = LaserTagPOMDP(robot_init=robs[i])
        p = solve(sol, pomdp)
        hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(i))
        up_rng = MersenneTwister(i+100_000)
        up = ObsAdaptiveParticleFilter(deepcopy(pomdp), LowVarianceResampler(10_000), 0.05, up_rng)
        hist = simulate(hr, pomdp, p, up)
        discounted_reward(hist)
    end
    @show k 
    @show mean(rewards)
end
