using ContinuousPOMDPTreeSearchExperiments
using Base.Test

using POMDPModels
using POMDPToolbox

pomdp = BabyPOMDP()
fwc = FeedWhenCrying()

q = []
push!(q, Sim(pomdp, fwc, max_steps=32, rng=MersenneTwister(4), metadata=Dict(:note=>"note")))
push!(q, Sim(pomdp, fwc, max_steps=32, rng=MersenneTwister(4)))

@show run_parallel(q) do sim, hist
    stuff = metadata_as_pairs(sim)
    push!(stuff, :steps=>n_steps(hist)) 
    push!(stuff, :reward=>discounted_reward(hist))
    return stuff
end

@show data = run_parallel(q)
@test data[1, :reward] == data[2, :reward]
