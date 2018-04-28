using ContinuousPOMDPTreeSearchExperiments
using ProgressMeter
using POMCPOW
using POMDPs
@everywhere using POMDPToolbox

m = LQG1D(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3)
filter = Kalman1D(m)
policy = LinearFeedback(1/2)

@show N = 1_000_000

#=
if !isdefined(:results)
    results = DataFrame()
end

if isempty(results)
    sims = []
    @showprogress for i in 1:N
        rng = MersenneTwister(i)
        sim = Sim(m, policy, filter, rng=rng)
        push!(sims, sim)
    end
    results = run_parallel(sims) do sim, hist
        sh = state_hist(hist)
        ah = action_hist(hist)
        oh = observation_hist(hist)
        @assert length(sh) == 3
        @assert length(ah) == 2
        return [:x1=>sh[1].x,
                :x2=>sh[2].x,
                :x3=>sh[3].x,
                :a2=>ah[2],
                :z2=>oh[1],
                :reward=>discounted_reward(hist)]
    end
end

@show var(results[:x1])
@show var(results[:x2])
@show var(results[:x3])
@show var(results[:a2])
@show mean(results[:reward])
# @show results[:a2]./results[:z2]
=#

# @time rsum = @parallel (+) for i in 1:N
#     rng = MersenneTwister(i+5_000_000)
#     ro = RolloutSimulator(rng=rng)
#     simulate(ro, m, policy, filter)
# end
# @show rsum/N

ps = POMCPOWSolver(estimate_value=FORollout(policy),
                   k_action=2.0,
                   k_observation=2.0,
                   alpha_action=1/8.0,
                   alpha_observation=1/8.0,
                   check_repeat_act=true,
                   check_repeat_obs=false,
                   criterion=MaxUCB(10.0)
                  )
planner = solve(ps, m)
b = initial_state_distribution(m)

rng = MersenneTwister(7)

tree = POMCPOW.make_tree(planner, b)
@showprogress for i in 1:1_000_000
    POMCPOW.simulate(planner, POWTreeObsNode(tree, 1), rand(rng, b), 3)
end

best_node = POMCPOW.select_best(MaxTries(), POWTreeObsNode(tree, 1), Base.GLOBAL_RNG)
@show tree.v[best_node]
