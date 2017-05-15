using VDPTag
using Plots
using POMDPToolbox
using Reel
using ProgressMeter
using PmapProgressMeter
using MCTS
using Plots

@show N = 1000

@everywhere begin
    using POMDPs
    using POMDPToolbox
    using MCTS
    using VDPTag

    mdp = VDPTagMDP()
    dmdp = DiscreteVDPTagMDP()

    n = 10000
    seed = 42

    planners = Dict{String, Policy}(
        "continuous_dpw" => begin
            solver = DPWSolver(depth=20,
                               exploration_constant=40.0,
                               rng=MersenneTwister(seed),
                               estimate_value=RolloutEstimator(ToNextML(mdp)),
                               n_iterations=n,
                               k_action=8.0,
                               alpha_action=1/20,
                               k_state=4.0,
                               alpha_state=1/20)
            solve(solver, mdp)
        end,

        "discrete_dpw" => begin
            ro = translate_policy(ToNextML(mdp), mdp, dmdp, dmdp)
            solver = DPWSolver(depth=20,
                               exploration_constant=40.0,
                               rng=MersenneTwister(seed),
                               estimate_value=RolloutEstimator(ro),
                               n_iterations=n,
                               k_action=8.0,
                               alpha_action=1/20,
                               k_state=4.0,
                               alpha_state=1/20)
            translate_policy(solve(solver, dmdp), dmdp, mdp, dmdp)
        end,

        "discrete_mcts" => begin
            ro = translate_policy(ToNextML(mdp), mdp, dmdp, dmdp)
            solver = MCTSSolver(depth=20,
                                exploration_constant=40.0,
                                rng=MersenneTwister(seed),
                                estimate_value=RolloutEstimator(ro),
                                n_iterations=n)
            translate_policy(solve(solver, dmdp), dmdp, mdp, dmdp)
        end,

        "discrete_random" => translate_policy(RandomPolicy(dmdp, rng=MersenneTwister(seed)), dmdp, mdp, dmdp),

        "discrete_heur" => begin
            dheur = translate_policy(ToNextML(mdp), mdp, dmdp, dmdp)
            translate_policy(dheur, dmdp, mdp, dmdp)
        end,

        "continuous_heur" => ToNextML(mdp)
    )
end

for (k,p) in planners
    prog = Progress(N, desc="Simulating...")
    rewards = pmap(prog, 1:N) do i
        hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(i))
        hist = simulate(hr, mdp, p)
        discounted_reward(hist)
    end
    @show k 
    @show mean(rewards)
end
