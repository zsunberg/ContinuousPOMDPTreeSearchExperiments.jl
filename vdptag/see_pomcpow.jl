using VDPTag
using Plots
using POMDPToolbox
using Reel
using ProgressMeter
using PmapProgressMeter
using ContinuousPOMDPTreeSearchExperiments
using MCTS
using ParticleFilters
using POMCPOW
using POMCP

@everywhere begin
    using POMDPs
    using POMDPToolbox
    using MCTS
    using VDPTag
    using POMCPOW
    using POMCP
    using ContinuousPOMDPTreeSearchExperiments
    using ParticleFilters

    pomdp = VDPTagPOMDP()
    seed=27

    rollout_policy = ToNextML(mdp(pomdp))
    solver = POMCPOWSolver(tree_queries=100_000,
                           criterion=MaxUCB(40.0),
                           final_criterion=MaxTries(),
                           max_depth=20,
                           k_action=8.0,
                           alpha_action=1/20,
                           k_observation=4.0,
                           alpha_observation=1/20,
                           estimate_value=FORollout(rollout_policy),
                           # check_repeat_act=false,
                           # check_repeat_obs=false,
                           rng=MersenneTwister(seed)
                          )

    updater = ObsAdaptiveParticleFilter(pomdp, LowVarianceResampler(100_000), 0.0, MersenneTwister(seed+100))
end


N = 100
# s_rewards = SharedArray(Float64, N)
prog = Progress(N, desc="Simulating...")
rewards = pmap(prog, 1:N) do i
# @showprogress "Simulating..." for i in 1:N
    policy = solve(solver, pomdp)

    hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(i))
    hist = simulate(hr, pomdp, policy, updater)
    discounted_reward(hist)
end
@show mean(rewards)

#=
policy = solve(solver, pomdp)
action(policy, initial_state_distribution(pomdp))
blink(policy)
=#

#=
policy = solve(solver, pomdp)

hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(1), show_progress=true)
hist = simulate(hr, pomdp, policy, updater)
@show discounted_reward(hist)

gr()
frames = Frames(MIME("image/png"), fps=2)
@showprogress "Creating gif..." for i in 1:length(hist)
    push!(frames, plot(pomdp, view(hist, 1:i)))
end

filename = string(tempname(), "_vdprun.gif")
write(filename, frames)
println(filename)
run(`setsid gifview $filename`)
=#
