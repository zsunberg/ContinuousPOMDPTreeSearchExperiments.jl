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

    mdp = VDPTagMDP()
    seed=27

    solver = DPWSolver(depth=20,
                       exploration_constant=40.0,
                       rng=MersenneTwister(seed),
                       estimate_value=RolloutEstimator(ToNextML(mdp)),
                       n_iterations=1000,
                       k_action=8.0,
                       alpha_action=1/20,
                       k_state=4.0,
                       alpha_state=1/20)
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
policy = solve(solver, mdp)

hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(1))
hist = simulate(hr, mdp, policy)
@show discounted_reward(hist)

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
