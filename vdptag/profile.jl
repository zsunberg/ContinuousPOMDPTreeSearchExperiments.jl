using POMDPs
using ContinuousPOMDPTreeSearchExperiments
using POMDPToolbox
using POMCPOW
using VDPTag
using MCTS
using DataFrames
using ParticleFilters
using ProfileView

pomdp = VDPTagPOMDP()

#=
rng = MersenneTwister(13)
rollout_policy = ToNextML(mdp(pomdp))
node_updater = ObsAdaptiveParticleFilter(pomdp, LowVarianceResampler(100), 0.05, rng)
solver = DPWSolver(n_iterations=10_000_000,
                   exploration_constant=40.0,
                   max_time=1.0,
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
planner = solve(solver, belief_mdp)
=#

rollout_policy = ToNextML(mdp(pomdp))
solver = POMCPOWSolver(tree_queries=10_000_000,
                       criterion=MaxUCB(40.0),
                       final_criterion=MaxTries(),
                       max_depth=20,
                       max_time=1.0,
                       k_action=8.0,
                       alpha_action=1/20,
                       k_observation=4.0,
                       alpha_observation=1/20,
                       estimate_value=FORollout(rollout_policy),
                       check_repeat_act=false,
                       check_repeat_obs=false,
                       rng=MersenneTwister(13)
                      )
planner = solve(solver, deepcopy(pomdp))

up = ObsAdaptiveParticleFilter(deepcopy(pomdp), LowVarianceResampler(10_000), 0.05, MersenneTwister(90_000))

b = initialize_belief(up, initial_state_distribution(pomdp))
@time action(planner, b)
@time action(planner, b)
@time action(planner, b)

Profile.clear()
@profile action(planner, b)
ProfileView.view()
