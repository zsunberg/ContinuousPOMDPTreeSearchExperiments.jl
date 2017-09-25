using LaserTag
using Plots
using POMDPToolbox
using Reel
using ProgressMeter
using PmapProgressMeter
using ParticleFilters
using ContinuousPOMDPTreeSearchExperiments
using Plots
using QMDP
using JLD
# using DESPOT
using BasicPOMCP
using ARDESPOT

file_contents = readstring(@__FILE__())

@everywhere begin
    using POMDPs
    using POMDPToolbox
    using ContinuousPOMDPTreeSearchExperiments
    using DiscreteValueIteration
    using ParticleFilters
    using POMCPOW
    using LaserTag
    using QMDP
    # using DESPOT
    using BasicPOMCP
    using ARDESPOT

    N = 100
    n = 1_000_000
    P = typeof(gen_lasertag(rng=MersenneTwister(18)))
    lambda = 0.0

    solvers = Dict{String, Union{Policy, Solver}}(

        #=
        "pomcpow" => begin
            # ro = MoveTowards()
            solver = POMCPOWSolver(tree_queries=n, #500_000
                                   criterion=MaxUCB(40.0),
                                   final_criterion=MaxTries(),
                                   max_depth=100,
                                   enable_action_pw=false,
                                   # k_action=4.0,
                                   # alpha_action=1/8,
                                   k_observation=4.0,
                                   alpha_observation=1/20,
                                   estimate_value=FOValue(ValueIterationSolver()),
                                   check_repeat_act=false,
                                   check_repeat_obs=false,
                                   init_N=InevitableInit(),
                                   init_V=InevitableInit(),
                                   rng=MersenneTwister(13)
                                  )
            solver
        end,
        =#

        # "move_towards_sampled" => MoveTowardsSampled(MersenneTwister(17)),

        "qmdp" => QMDPSolver(max_iterations=1000),

        # "ml" => OptimalMLSolver(ValueIterationSolver()),

        # "be" => BestExpectedSolver(ValueIterationSolver()),

        # "random" => RandomSolver(rng=MersenneTwister(4)),

        "despot" => begin
            DESPOTSolver(lambda=lambda,
                         max_trials=1_000_000,
                         T_max=5.0,
                         bounds=LaserBounds{P}(),
                         rng=MersenneTwister(4))
        end,

        #=
        "pomcp" => POMCPSolver(tree_queries=n,
                                   c=20.0,
                                   max_depth=100,
                                   estimate_value=FOValue(ValueIterationSolver()),
                                   rng=MersenneTwister(13)
                                  )
        =#
    )
end

@show N
@show n
@show lambda

rdict = Dict{String, Any}()
for (k,sol) in solvers
    prog = Progress(N, desc="Simulating...")
    @show k 
    rewards = pmap(prog, 1:N) do i
    # rewards = map(1:N) do i
        pomdp = gen_lasertag(rng=MersenneTwister(i+600_000))
        if isa(sol,Solver)
            p = solve(deepcopy(sol), pomdp)
        else
            p = sol
        end
        hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(i))
        up_rng = MersenneTwister(i+100_000)
        up = ObsAdaptiveParticleFilter(deepcopy(pomdp), LowVarianceResampler(100_000), 0.05, up_rng)
        hist = simulate(hr, pomdp, p, up)
        discounted_reward(hist)
    end
    @show mean(rewards)
    @show std(rewards)/sqrt(N)
    rdict[k] = rewards
end

filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "data", "laser_discrete_run_$(Dates.format(now(), "E_d_u_HH_MM")).jld")
println("saving to $filename...")
@save(filename, solvers, rdict, file_contents)
println("done.")
