using VDPTag
using Plots
using POMDPToolbox
using Reel
using ProgressMeter
using PmapProgressMeter
using MCTS

@everywhere begin
    using POMDPs
    using POMDPToolbox
    using MCTS
    using VDPTag

    pomdp = VDPTagPOMDP()
    seed=27

    rollout_policy = ToNextML(mdp(pomdp))
    solver = POMCPOWSolver(tree_queries=10_000,
                           criterion=MaxUCB(20),
                           final_criterion=MaxTries(),
                           max_depth=20,
                           k_action=8.0,
                           alpha_action=1/4,
                           k_observation=8.0,
                           alpha_observation=1/4,
                           estimate_value=FORollout(rollout_policy),
                           rng=rng3
                          )
end


N = 100
# s_rewards = SharedArray(Float64, N)
prog = Progress(N, desc="Simulating...")
rewards = pmap(prog, 1:N) do i
# @showprogress "Simulating..." for i in 1:N
    policy = solve(solver, mdp)

    hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(i))
    hist = simulate(hr, mdp, policy)
    discounted_reward(hist)
end
@show mean(rewards)

#=
gr()
frames = Frames(MIME("image/png"), fps=2)
@showprogress "Creating gif..." for s in state_hist(hist)
    push!(frames, plot(mdp, s))
end

filename = string(tempname(), "_vdprun.gif")
write(filename, frames)
println(filename)
run(`setsid gifview $filename`)
=#
