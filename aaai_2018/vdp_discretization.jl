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
using DataFrames

N = 4

pomdp = VDPTagPOMDP()

solvers = Dict{String, Union{Solver,Policy}}(

    "pomcpow" => begin
        ro = ToNextML(mdp(pomdp))
        solver = POMCPOWSolver(tree_queries=10_000_000,
                               criterion=MaxUCB(40.0),
                               final_criterion=MaxTries(),
                               max_depth=20,
                               max_time=0.1,
                               k_action=8.0,
                               alpha_action=1/20,
                               k_observation=4.0,
                               alpha_observation=1/20,
                               estimate_value=0.0,
                               check_repeat_act=true,
                               check_repeat_obs=true,
                               default_action=1,
                               rng=MersenneTwister(13)
                              )
    end,

    "pomcp" => begin
        ro = ToNextML(mdp(pomdp))
        POMCPSolver(max_depth=20,
                    max_time=0.1,
                    c=40.0,
                    tree_queries=typemax(Int),
                    default_action=1,
                    estimate_value=0.0,
                    rng=MersenneTwister(17)
                   )
    end
)

alldata = DataFrame()
for n_angles_float in logspace(0.5, 3, 6)
    n_angles = round(Int, n_angles_float)
    for (k, solver) in solvers
        println("$k ($n_angles)")
        dpomdp = AODiscreteVDPTagPOMDP(n_angles=n_angles, n_obs_angles=n_angles)
        planner = solve(solver, dpomdp)
        sims = []
        for i in 1:N
            srand(planner, i+50_000)
            cplanner = translate_policy(planner, dpomdp, pomdp, dpomdp)

            filter = ObsAdaptiveParticleFilter(deepcopy(pomdp),
                                               LowVarianceResampler(10_000),
                                               0.05, MersenneTwister(i+90_000))            

            sim = Sim(deepcopy(pomdp),
                      deepcopy(cplanner),
                      filter,
                      rng=MersenneTwister(i+70_000),
                      max_steps=100,
                      metadata=Dict(:solver=>k,
                                    :n_angles=>n_angles,
                                    :i=>i)
                     )

            push!(sims, sim)
        end

        data = run_parallel(sims)

        rs = data[:reward]
        println(@sprintf("reward: %6.3f ± %6.3f", mean(rs), std(rs)/sqrt(length(rs))))
        alldata = vcat(alldata, data)
    end
end

#=
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
=#
