using VDPTag
using Plots
using POMDPToolbox
using Reel
using ProgressMeter
using PmapProgressMeter
using MCTS
using Plots

@everywhere begin
    using POMDPs
    using POMDPToolbox
    using MCTS
    using VDPTag
    using PmapProgressMeter
end

n_points = 8
mrew = Array(Float64, n_points)
ns = Array(Float64, n_points)
for j = 1:n_points

    N = 1000
    # s_rewards = SharedArray(Float64, N)
    prog = Progress(N, desc="Simulating...")
    rewards = pmap(prog, 1:N, fill(j,N)) do i, j
    # @showprogress "Simulating..." for i in 1:N
        mdp = VDPTagMDP()
        seed = 27
        n = floor(Int, 10^(j/2))
        solver = DPWSolver(depth=20,
                           exploration_constant=40.0,
                           rng=MersenneTwister(seed),
                           estimate_value=RolloutEstimator(ToNextML(mdp)),
                           n_iterations=n,
                           k_action=8.0,
                           alpha_action=1/20,
                           k_state=4.0,
                           alpha_state=1/20)

        policy = solve(solver, mdp)

        hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(i))
        hist = simulate(hr, mdp, policy)
        discounted_reward(hist)
    end
    @show ns[j] = 10^(j/2)
    @show mrew[j] = mean(rewards)
end

unicodeplots()
plot(ns, mrew)
gui()
