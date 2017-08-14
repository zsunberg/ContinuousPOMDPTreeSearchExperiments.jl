using LaserTag
using POMDPToolbox
using ParticleFilters
using ContinuousPOMDPTreeSearchExperiments
using QMDP
using JLD
using DESPOT
using BasicPOMCP
using StaticArrays
using POMDPs
using Reel

f = ARGS[1]
k = ARGS[2]
i = parse(Int, ARGS[3])

rdict = load(f, "rdict")

n = 100_000

sol = DESPOTSolver{LTState, Int, DMeas, LaserBounds,
                      MersenneStreamArray}(bounds = LaserBounds{LaserTagPOMDP{DESPOTEmu, DMeas}}(),
                                           random_streams=MersenneStreamArray(MersenneTwister(1)),
                                           rng=MersenneTwister(3),
                                           next_state=LTState([1,1], [1,1], false),
                                           curr_obs=DMeas(),
                                           time_per_move=100.0,
                                           eta=0.01,
                                           max_trials=n # 500_000
                                          )

pomdp = gen_lasertag(rng=MersenneTwister(i+600_000))
if isa(sol,Solver)
    p = solve(deepcopy(sol), pomdp)
else
    p = sol
end
hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(i))
up_rng = MersenneTwister(i+100_000)
# up = ObsAdaptiveParticleFilter(deepcopy(pomdp), LowVarianceResampler(10_000), 0.05, up_rng)
up = ObsAdaptiveParticleFilter(deepcopy(pomdp), LowVarianceResampler(100_000), 0.05, up_rng)

frames = Frames(MIME("image/png"), fps=2)
disc = 1.0
rew = 0.0
for (s, b, a, r, sp, o, bp) in stepthrough(pomdp, p, up, "sbarspobp", max_steps=100, rng=MersenneTwister(i))
    rew += disc*r
    disc*=discount(pomdp)
    show(STDOUT, MIME("text/plain"), LaserTagVis(pomdp, s=s))
    println()
    push!(frames, LaserTagVis(pomdp, s=sp, b=bp, o=o, a=a))
end

println("Total discounted reward for this simulation: $rew")
println("Total discounted reward from original: $(rdict[k][i])")

gifname = "out"*randstring()*".gif"
println("Writing $gifname...")
write(gifname, frames)
println("done.")
