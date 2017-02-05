using LightDarkPOMDPs
using POMDPToolbox
using POMCP
using Plots
using Parameters
using StaticArrays
using POMDPs
using ContinuousPOMDPTreeSearchExperiments
using ParticleFilters
using Reel

pomdp = LightDark2DTarget(term_radius=0.1,
                    init_dist=SymmetricNormal2([2.0, 2.0], 5.0),
                    discount=0.95)

rng_seed = 5
rng = MersenneTwister(rng_seed)

## VANILLA ##

#=
rng3 = copy(rng)
action_gen = AdaptiveRadiusRandom(max_radius=10.0, to_zero_first=true, rng=rng3)
rollout_policy = RadiusRandom(radius=10.0, rng=rng3)
tree_queries = 50_000
updater = ObsAdaptiveParticleFilter(pomdp, LowVarianceResampler(1000), 0.05, rng)
solver = POMCPDPWSolver(next_action=action_gen,
                        tree_queries=tree_queries,
                        c=20.0,
                        max_depth=40,
                        k_action=10.0,
                        alpha_action=1/8,
                        k_observation=4.0,
                        alpha_observation=1/8,
                        estimate_value=RolloutEstimator(rollout_policy),
                        # estimate_value=PORolloutEstimator(rollout_policy, node_updater),
                        # node_belief_updater=node_updater,
                        rng=rng3
                       )
policy = solve(solver, pomdp)

#=
ib = initial_state_distribution(pomdp)
node = RootNode(ib)
a = action(policy, node)
POMCP.blink(node)
=#

hr = HistoryRecorder(max_steps=40, rng=MersenneTwister(rng_seed), initial_state=Vec2(2.0, 3.0), show_progress=true)
hist = simulate(hr, pomdp, policy, updater)
println(discounted_reward(hist))

steps = length(hist)
film = roll(fps=1, duration=steps-1) do t, dt
    print(".")
    v = view(hist, 1:Int(t+1))
    plot(pomdp)
    plot!(v)
    b = belief_hist(v)[end]
    plot!(b)
end
filename = string(tempname(), "Vanilla_$(tree_queries)_$rng_seed.gif")
write(filename, film)
println(filename)
run(`setsid gifview $filename`)
=#


## WITH FEEDBACK ##

rng3 = copy(rng)
action_gen = AdaptiveRadiusRandom(max_radius=10.0, to_zero_first=true, rng=rng3)
rollout_policy = SimpleFeedback(max_radius=10.0)
rollout_updater = FastPreviousObservationUpdater{Vec2}()
tree_queries = 50_000
updater = ObsAdaptiveParticleFilter(pomdp, LowVarianceResampler(1000), 0.05, rng)
solver = POMCPDPWSolver(next_action=action_gen,
                        tree_queries=tree_queries,
                        c=20.0,
                        max_depth=40,
                        k_action=10.0,
                        alpha_action=1/8,
                        k_observation=4.0,
                        alpha_observation=1/8,
                        estimate_value=PORolloutEstimator(rollout_policy, rollout_updater),
                        rng=rng3
                       )
policy = solve(solver, pomdp)

#=
ib = initial_state_distribution(pomdp)
node = RootNode(ib)
a = action(policy, node)
POMCP.blink(node)
=#

hr = HistoryRecorder(max_steps=40, rng=MersenneTwister(rng_seed), initial_state=Vec2(2.0, 3.0), show_progress=true)
hist = simulate(hr, pomdp, policy, updater)
println(discounted_reward(hist))

steps = length(hist)
film = roll(fps=1, duration=steps) do t, dt
    print(".")
    v = view(hist, 1:Int(t+1))
    plot(pomdp)
    plot!(v)
    b = belief_hist(v)[end]
    plot!(b)
end
filename = string(tempname(), "Feedback_$(tree_queries)_$rng_seed.gif")
write(filename, film)
println(filename)
run(`setsid gifview $filename`)


## WITH BELIEF ##

#=
rng3 = copy(rng)
action_gen = AdaptiveRadiusRandom(max_radius=10.0, to_zero_first=true, rng=rng3)
rollout_policy = SimpleFeedback(max_radius=10.0)
tree_queries = 50_000
updater = ObsAdaptiveParticleFilter(pomdp, LowVarianceResampler(1000), 0.05, rng)
node_updater = ObsAdaptiveParticleFilter(pomdp, LowVarianceResampler(100), 0.05, rng)
solver = POMCPDPWSolver(next_action=action_gen,
                        tree_queries=tree_queries,
                        c=20,
                        max_depth=40,
                        k_action=10.0,
                        alpha_action=1/8,
                        k_observation=4.0,
                        alpha_observation=1/8,
                        estimate_value=PORolloutEstimator(rollout_policy, node_updater),
                        node_belief_updater=node_updater,
                        rng=rng3
                       )
policy = solve(solver, pomdp)

#=
ib = initial_state_distribution(pomdp)
node = RootNode(ib)
a = action(policy, node)
POMCP.blink(node)
=#

hr = HistoryRecorder(max_steps=40, rng=MersenneTwister(rng_seed), initial_state=Vec2(2.0, 3.0), show_progress=true)
hist = simulate(hr, pomdp, policy, updater)
println(discounted_reward(hist))

steps = length(hist)
film = roll(fps=1, duration=steps) do t, dt
    print(".")
    v = view(hist, 1:Int(t+1))
    plot(pomdp)
    plot!(v)
    b = belief_hist(v)[end]
    plot!(b)
end
filename = string(tempname(), "With_Belief_Random_Rollout_$(tree_queries)_$rng_seed.gif")
write(filename, film)
println(filename)
run(`setsid gifview $filename`)
=#
