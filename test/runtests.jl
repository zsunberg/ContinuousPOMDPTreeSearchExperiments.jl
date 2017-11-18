using ContinuousPOMDPTreeSearchExperiments
using Base.Test

using POMDPModels
using POMDPToolbox
using LaserTag
using ParticleFilters
using QMDP

srand(4)

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

sld = SimpleLightDark()
@test isterminal(sld, sld.radius+1)
p = solve(LDHSolver(), sld)
filter = SIRParticleFilter(sld, 1000)
for (s, b, a, r, sp, o) in stepthrough(sld, p, filter, "sbarspo", max_steps=100)
    @show (s, a, r, sp, o)
    @show mean(b)
end

qp = solve(QMDPSolver(), sld, verbose=true)
filter = SIRParticleFilter(sld, 1000)
for (s, b, a, r, sp, o) in stepthrough(sld, qp, filter, "sbarspo", max_steps=100)
    @show (s, a, r, sp, o)
    @show mean(b)
end
