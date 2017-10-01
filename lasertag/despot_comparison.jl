using LaserTag
using Plots
using POMDPToolbox
using Reel
using ProgressMeter
using PmapProgressMeter
using ParticleFilters
using ContinuousPOMDPTreeSearchExperiments
using Plots
using QMDP
using JLD
# using DESPOT
using BasicPOMCP
using ARDESPOT

file_contents = readstring(@__FILE__())

# @everywhere begin
begin
    using POMDPs
    using POMDPToolbox
    using ContinuousPOMDPTreeSearchExperiments
    using DiscreteValueIteration
    using ParticleFilters
    using POMCPOW
    using LaserTag
    using QMDP
    import DESPOT
    using BasicPOMCP
    using ARDESPOT

    N = 1
    P = typeof(gen_lasertag(rng=MersenneTwister(18)))

    solvers = Dict{String, Union{Policy, Solver}}(

        "qmdp" => QMDPSolver(max_iterations=1000),

        "ardespot" => begin
            DESPOTSolver(lambda=0.0,
                         max_trials=1_000_000,
                         T_max=5.0,
                         bounds=LaserBounds{P}(),
                         default_action=nogap_tag,
                         rng=MersenneTwister(4))
        end,

        "despot" => begin
            DESPOT.DESPOTSolver{LTState,
                                Int,
                                DMeas,
                                LaserBounds{P},
                                DESPOT.MersenneStreamArray}(
                bounds=LaserBounds{P}(),
                time_per_move=5.0,
                max_trials=1_000_000,
                random_streams=DESPOT.MersenneStreamArray(MersenneTwister(4)),
                next_state=LTState([-1,-1],[-1,-1], false)
            )
        end
    )
end

@show N

rdict = Dict{String, Any}()
for (k,sol) in solvers
    prog = Progress(N, desc="Simulating...")
    @show k 
    # rewards = pmap(prog, 1:N) do i
    rewards = map(1:N) do i
        pomdp = gen_lasertag(rng=MersenneTwister(i+600_000))
        if isa(sol,Solver)
            p = solve(deepcopy(sol), pomdp)
        else
            p = sol
        end
        hr = HistoryRecorder(max_steps=100, rng=MersenneTwister(i))
        up_rng = MersenneTwister(i+100_000)
        up = ObsAdaptiveParticleFilter(deepcopy(pomdp), LowVarianceResampler(100_000), 0.05, up_rng)
        hist = simulate(hr, pomdp, p, up)
        discounted_reward(hist)
    end
    @show mean(rewards)
    @show std(rewards)/sqrt(N)
    rdict[k] = rewards
end

filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "data", "despot_comparison_run_$(Dates.format(now(), "E_d_u_HH_MM")).jld")
println("saving to $filename...")
@save(filename, rdict, file_contents)
println("done.")
