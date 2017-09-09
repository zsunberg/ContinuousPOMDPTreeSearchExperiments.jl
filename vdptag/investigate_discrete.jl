using POMDPs
using ContinuousPOMDPTreeSearchExperiments
using POMDPToolbox
using BasicPOMCP
using POMCPOW
using ParticleFilters
using CPUTime
using VDPTag
using DataFrames
using ProfileView

pomdp = VDPTagPOMDP()

solvers = Dict{String, Union{Solver,Policy}}(

    "pomcpow" => begin
        ro = ToNextML(mdp(pomdp))
        solver = POMCPOWSolver(tree_queries=10_000_000,
                               criterion=MaxUCB(40.0),
                               final_criterion=MaxTries(),
                               max_depth=20,
                               max_time=0.1,
                               k_action=8.0,
                               alpha_action=1/20,
                               k_observation=4.0,
                               alpha_observation=1/20,
                               estimate_value=0.0,
                               check_repeat_act=true,
                               check_repeat_obs=true,
                               rng=MersenneTwister(13)
                              )
    end,

    "pomcp" => begin
        ro = ToNextML(mdp(pomdp))
        POMCPSolver(max_depth=20,
                    max_time=0.1,
                    c=40.0,
                    tree_queries=typemax(Int),
                    # default_action=TagAction(false, 0.0),
                    estimate_value=0.0,
                    rng=MersenneTwister(17)
                   )
    end
)

n_angles = 10
dpomdp = AODiscreteVDPTagPOMDP(n_angles=n_angles, n_obs_angles=n_angles)

solver = solvers["pomcp"]

planner = solve(solver, dpomdp)
cplanner = translate_policy(planner, dpomdp, pomdp, dpomdp)

filter = ObsAdaptiveParticleFilter(deepcopy(pomdp),
                                   LowVarianceResampler(10_000),
                                   0.05, MersenneTwister(90_000))            

@show simulate(RolloutSimulator(max_steps=100), pomdp, cplanner, filter)

Profile.clear()
@profile simulate(RolloutSimulator(max_steps=100), pomdp, cplanner, filter)
ProfileView.view()
