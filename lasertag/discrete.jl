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
using DESPOT


@everywhere begin
    using POMDPs
    using POMDPToolbox
    using ContinuousPOMDPTreeSearchExperiments
    using DiscreteValueIteration
    using ParticleFilters
    using POMCPOW
    using POMCP
    using LaserTag
    using QMDP
    using DESPOT

    N = 100

    solvers = Dict{String, Union{Policy, Solver}}(

    #=
        "pomcpow" => begin
            # ro = MoveTowards()
            solver = POMCPOWSolver(tree_queries=500_000,
                                   criterion=MaxUCB(20.0),
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

        "ml" => OptimalMLSolver(ValueIterationSolver()),

        "be" => BestExpectedSolver(ValueIterationSolver()),

        #=
        "despot" => DESPOTSolver{LTState,
                      Int,
                      DMeas,
                      LaserBounds,
                      MersenneStreamArray}(bounds = LaserBounds{DMeas}(),
                                           random_streams=MersenneStreamArray(MersenneTwister(1)),
                                           rng=MersenneTwister(3),
                                           next_state=LTState([1,1], [1,1], false),
                                           curr_obs=DMeas(),
                                           time_per_move=-1.0,
                                           max_trials=500_000
                                          ),
        =#

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

rdict = Dict{String, Any}()
for (k,sol) in solvers
    prog = Progress(N, desc="Simulating...")
    @show k 
    rewards = pmap(prog, 1:N) do i
        pomdp = gen_lasertag(rng=MersenneTwister(i+600_000), discrete=true)
        if isa(sol,Solver)
            p = solve(deepcopy(sol), pomdp)
        else
            p = sol
        end
        hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(i))
        up_rng = MersenneTwister(i+100_000)
        up = ObsAdaptiveParticleFilter(deepcopy(pomdp), LowVarianceResampler(10_000), 0.05, up_rng)
        hist = simulate(hr, pomdp, p, up)
        discounted_reward(hist)
    end
    @show mean(rewards)
    rdict[k] = rewards
end

filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "data", "laser_pomcpow_run_$(Dates.format(now(), "E_d_u_HH_MM")).jld")
println("saving to $filename...")
@save(filename, solvers, rdict)
println("done.")
