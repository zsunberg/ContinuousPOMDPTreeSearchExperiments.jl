using POMDPs
using ContinuousPOMDPTreeSearchExperiments
using POMDPToolbox
using BasicPOMCP
using POMCPOW
using ProgressMeter
using PmapProgressMeter
using ParticleFilters
using JLD
using CPUTime
using VDPTag

N = 100

@everywhere begin
    using VDPTag
    using POMDPs
    using ContinuousPOMDPTreeSearchExperiments
    using POMDPToolbox
    using POMCP
    using POMCPOW
    using ParticleFilters
    using CPUTime

    pomdp = VDPTagPOMDP()
    p_rng = MersenneTwister(4)
    solvers = Dict{String, Union{Solver,Policy}}(

        "pomcpow" => begin
            rollout_policy = ToNextML(mdp(pomdp))
            solver = POMCPOWSolver(tree_queries=10_000,
                                   criterion=MaxUCB(40.0),
                                   final_criterion=MaxTries(),
                                   max_depth=20,
                                   k_action=8.0,
                                   alpha_action=1/20,
                                   k_observation=4.0,
                                   alpha_observation=1/20,
                                   estimate_value=FORollout(rollout_policy),
                                   check_repeat_act=false,
                                   check_repeat_obs=false,
                                   rng=MersenneTwister(13)
                                  )

        end

    )
end

solver_keys = collect(keys(solvers))
rewards = Dict{String, AbstractVector{Float64}}()
# state_hists = Dict{String, AbstractVector{AbstractVector{state_type(pomdp)}}}()
times = Dict{String, AbstractVector{Float64}}()
steps = Dict{String, AbstractVector{Int}}()

for (j, sk) in enumerate(solver_keys)
    s_rewards = SharedArray(Float64, N) 
    # s_hists = SharedArray(AbstractVector{state_type(pomdp)}, N)
    s_times = SharedArray(Float64, N)
    s_steps = SharedArray(Float64, N)
    prog = Progress(N, desc="$sk ($j of $(length(solver_keys)))...")
    pmap(prog, 1:N) do i 
        sim_rng = MersenneTwister(i)
        up_rng = MersenneTwister(i+100_000)
        policy = solve(solvers[sk], pomdp)
        up = ObsAdaptiveParticleFilter(deepcopy(pomdp), LowVarianceResampler(10_000), 0.05, up_rng)
        sim = HistoryRecorder(max_steps=40, rng=sim_rng)
        s_times[i] = @CPUelapsed hist = simulate(sim, deepcopy(pomdp), policy, up)
        s_rewards[i] = discounted_reward(hist)
        s_steps[i] = n_steps(hist)
    end
    rewards[sk] = sdata(s_rewards)
    @show mean(rewards[sk])
    # state_hists[sk] = sdata(s_hists)
    times[sk] = sdata(s_times)
    steps[sk] = sdata(s_steps)
    @show sum(times[sk])/sum(steps[sk])
end

for k in solver_keys
    println("$k mean: $(mean(rewards[k])) sem: $(std(rewards[k])/sqrt(N))")
    println("$k time per step: $(sum(times[k])/sum(steps[k]))")
end

filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "data", "compare_$(Dates.format(now(), "E_d_u_HH_MM")).jld")
println("saving to $filename...")
@save(filename, solver_keys, rewards, times, steps)
println("done.")
