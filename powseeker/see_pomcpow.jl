using Powseeker
using POMDPToolbox
using POMCP
using POMCPOW
using Plots
using Parameters
using POMDPs
using ContinuousPOMDPTreeSearchExperiments
using ParticleFilters
using Reel
using ProgressMeter

# using Gallium
# Gallium.breakpoint(Pkg.dir("ContinuousPOMDPTreeSearchExperiments","src","ContinuousPOMDPTreeSearchExperiments.jl"), 68)
# @conditional Gallium.breakpoint(Pkg.dir("POMCPOW","src","solver2.jl"), 112) (h_node.node==14)

pomdp = PowseekerPOMDP()
# rng_seed = 4
rng_seed = 6
rng = MersenneTwister(rng_seed)



rng3 = copy(rng)
action_gen = GPSFirst(rng3)
rollout_policy = SkiOver(pomdp)
tree_queries = 10_000
updater = ObsAdaptiveParticleFilter(pomdp, LowVarianceResampler(10_000), 0.05, rng)
solver = POMCPOWSolver(next_action=action_gen,
                        tree_queries=tree_queries,
                        criterion=MaxUCB(exp(35)),
                        final_criterion=MaxTries(),
                        max_depth=mdp(pomdp).duration+2,
                        k_action=8.0,
                        alpha_action=1/20,
                        k_observation=8.0,
                        alpha_observation=1/10,
                        estimate_value=FORollout(rollout_policy),
                        # node_belief_updater=updater,
                        rng=rng3
                       )
policy = solve(solver, pomdp)

#=
ib = initial_state_distribution(pomdp)
a = action(policy, ib)
POMCP.blink(policy)
=#

#=
using ProfileView
ib = initial_state_distribution(pomdp)
a = action(policy, ib)
Profile.clear()
@profile a = action(policy, ib)
ProfileView.view()
=#

hr = HistoryRecorder(max_steps=mdp(pomdp).duration, rng=MersenneTwister(rng_seed), show_progress=true)
hist = simulate(hr, pomdp, policy, updater)
println(discounted_reward(hist))

# using JLD
# filename = tempname()*".jld"
# JLD.save(filename, "hist", hist, "solver", solver, "pomdp", pomdp)
# println("saved to "*filename)

gr()
frames = Frames(MIME("image/png"), fps=2)
@showprogress "Creating gif..." for i in 1:n_steps(hist)
    plot(pomdp)
    push!(frames, plot!(view(hist, 1:i)))
end

filename = string(tempname(), "_pomcpow_powseeker.gif")
write(filename, frames)
println(filename)
run(`setsid gifview $filename`)
