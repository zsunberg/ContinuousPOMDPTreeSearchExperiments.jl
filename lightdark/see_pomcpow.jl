using LightDarkPOMDPs
using POMDPToolbox
using POMCP
using POMCPOW
using Plots
using Parameters
using StaticArrays
using POMDPs
using ContinuousPOMDPTreeSearchExperiments
using ParticleFilters
using Reel

# using Gallium
# Gallium.breakpoint(Pkg.dir("ContinuousPOMDPTreeSearchExperiments","src","ContinuousPOMDPTreeSearchExperiments.jl"), 68)
# @conditional Gallium.breakpoint(Pkg.dir("POMCPOW","src","solver2.jl"), 112) (h_node.node==14)

pomdp = LightDark2DTarget(term_radius=0.1,
                    init_dist=SymmetricNormal2([2.0, 2.0], 3.0),
                    discount=0.95)

rng_seed = 4
rng = MersenneTwister(rng_seed)


## POMCPOW ##
struct OneStepValue end
POMCP.estimate_value(o::OneStepValue, pomdp::POMDP, s, h, steps) = reward(pomdp, s)

rng3 = copy(rng)
action_gen = AdaptiveRadiusRandom(max_radius=6.0, to_zero_first=true, to_light_second=true, rng=rng3)
rollout_policy = SimpleFeedback(max_radius=10.0)
tree_queries = 500_000
updater = ObsAdaptiveParticleFilter(pomdp, LowVarianceResampler(100_000), 0.05, rng)
solver = POMCPOWSolver(next_action=action_gen,
                        tree_queries=tree_queries,
                        criterion=MaxUCB(5.0),
                        final_criterion=MaxTries(),
                        max_depth=40,
                        k_action=10.0,
                        alpha_action=1/8,
                        k_observation=4.0,
                        alpha_observation=1/8,
                        estimate_value=OneStepValue(),
                        node_belief_updater=updater,
                        rng=rng3
                       )
#policy = solve(solver, pomdp)
policy = POMCPPlanner2(solver, pomdp)

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

hr = HistoryRecorder(max_steps=40, rng=MersenneTwister(rng_seed), initial_state=Vec2(2.0, 3.0), show_progress=true)
hist = simulate(hr, pomdp, policy, updater)
println(discounted_reward(hist))

using JLD
filename = tempname()*".jld"
JLD.save(filename, "hist", hist, "solver", solver, "pomdp", pomdp)
println("saved to "*filename)

pyplot()
steps = length(hist)
film = roll(fps=1, duration=steps) do t, dt
    print(".")
    v = view(hist, 1:Int(t+1))
    plot(pomdp)
    plot!(v)
    b = belief_hist(v)[end]
    plot!(b)
end
filename = string(tempname(), "POMCPOW_$(tree_queries)_$rng_seed.gif")
write(filename, film)
println(filename)
run(`setsid gifview $filename`)
