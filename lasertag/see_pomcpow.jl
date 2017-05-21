using LaserTag
using Plots
using POMDPToolbox
using ContinuousPOMDPTreeSearchExperiments
using Plots
using POMCPOW
using POMDPs


ro = MoveTowards()
solver = POMCPOWSolver(tree_queries=10_000,
                       criterion=MaxUCB(20.0),
                       final_criterion=MaxTries(),
                       max_depth=100,
                       enable_action_pw=false,
                       k_observation=4.0,
                       alpha_observation=1/20,
                       estimate_value=FORollout(ro),
                       check_repeat_act=false,
                       check_repeat_obs=false,
                       init_N=InevitableInit(),
                       init_V=InevitableInit(),
                       rng=MersenneTwister(13)
                      )

p = gen_lasertag(rng=MersenneTwister(4))

policy = solve(solver, p)

action(policy, initial_state_distribution(p))

# blink(policy)
