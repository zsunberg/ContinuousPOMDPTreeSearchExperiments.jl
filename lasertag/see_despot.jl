using LaserTag
using POMDPToolbox
using ContinuousPOMDPTreeSearchExperiments
using POMDPs
using ProfileView
using ParticleFilters
using ARDESPOT

p = gen_lasertag(rng=MersenneTwister(4))

solver = DESPOTSolver(T_max=Inf,
             lambda=0.01,
             max_trials=1,
             bounds=LaserBounds{typeof(p)}(),
             rng=MersenneTwister(4))


planner = solve(solver, p)

b0 = initial_state_distribution(p)
s0 = rand(MersenneTwister(5), b0)
println(LaserTagVis(p, s=s0))

tree = ARDESPOT.build_despot(planner, b0)

println(TreeView(tree, 1, 3))

#=
@time action(policy, initial_state_distribution(p))

Profile.clear()
@profile action(policy, initial_state_distribution(p))
ProfileView.view()
=#

# hr = HistoryRecorder(max_steps=5, show_progress=true)
# filter = SIRParticleFilter(p, 100_000)
# 
# simulate(hr, p, policy, filter)
