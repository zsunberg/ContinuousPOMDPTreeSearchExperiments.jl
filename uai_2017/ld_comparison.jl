using POMDPs
using ContinuousPOMDPTreeSearchExperiments
using POMDPToolbox
using POMCP
using POMCPOW
using LightDarkPOMDPs
using ProgressMeter
using PmapProgressMeter
using ParticleFilters
using JLD
using CPUTime

N = 500

@everywhere begin
    using LightDarkPOMDPs
    using POMDPs
    using ContinuousPOMDPTreeSearchExperiments
    using POMDPToolbox
    using POMCP
    using POMCPOW
    using ParticleFilters

    pomdp = LightDark2DTarget(term_radius=0.1,
                              init_dist=SymmetricNormal2([2.0, 2.0], 5.0),
                              discount=0.95)
    p_rng = MersenneTwister(4)
    solvers = Dict{String, Union{Solver,Policy}}(
        #=
        "heuristic" => begin
            homing_policy = SimpleFeedback(gain=1.0, max_radius=10.0)
            exploring_up = ObsAdaptiveParticleFilter(pomdp, LowVarianceResampler(1000), 0.05, copy(p_rng))
            homing_up = ObsAdaptiveParticleFilter(pomdp, LowVarianceResampler(1000), 0.0, copy(p_rng))
            InfoGatherHeur(5.0, 0.03, homing_policy, exploring_up, homing_up)
        end,

        "pomcpdpw" => begin
            rng3 = copy(p_rng)
            action_gen = AdaptiveRadiusRandom(max_radius=6.0, to_zero_first=true, to_light_second=true, rng=rng3)
            rollout_policy = RadiusRandom(radius=10.0, rng=rng3)
            tree_queries = 1_000_000
            solver = POMCPDPWSolver(next_action=action_gen,
                                    tree_queries=tree_queries,
                                    c=5.0,
                                    max_depth=40,
                                    k_action=10.0,
                                    alpha_action=1/8,
                                    k_observation=4.0,
                                    alpha_observation=1/8,
                                    estimate_value=OneStepValue(),
                                    default_action=Vec2(0.1, 0.1),
                                    rng=rng3
                                   )
        end,

        "bt_100_ro_100k" => begin
            rng3 = copy(p_rng)
            action_gen = AdaptiveRadiusRandom(max_radius=6.0, to_zero_first=true, to_light_second=true, rng=rng3)
            rollout_policy = SimpleFeedback(max_radius=10.0)
            tree_queries = 100_000
            node_updater = ObsAdaptiveParticleFilter(pomdp, LowVarianceResampler(100), 0.05, rng3)
            solver = POMCPDPWSolver(next_action=action_gen,
                                    tree_queries=tree_queries,
                                    c=5.0,
                                    max_depth=40,
                                    k_action=10.0,
                                    alpha_action=1/8,
                                    k_observation=4.0,
                                    alpha_observation=1/8,
                                    estimate_value=PORolloutEstimator(rollout_policy, node_updater),
                                    node_belief_updater=node_updater,
                                    default_action=Vec2(0.1, 0.1),
                                    rng=rng3
                                   )
        end,

        "bt_100_osv_100k" => begin
            rng3 = copy(p_rng)
            action_gen = AdaptiveRadiusRandom(max_radius=6.0, to_zero_first=true, to_light_second=true, rng=rng3)
            rollout_policy = SimpleFeedback(max_radius=10.0)
            tree_queries = 100_000
            node_updater = ObsAdaptiveParticleFilter(pomdp, LowVarianceResampler(100), 0.05, rng3)
            solver = POMCPDPWSolver(next_action=action_gen,
                                    tree_queries=tree_queries,
                                    c=5.0,
                                    max_depth=40,
                                    k_action=10.0,
                                    alpha_action=1/8,
                                    k_observation=4.0,
                                    alpha_observation=1/8,
                                    estimate_value=OneStepValue(),
                                    node_belief_updater=node_updater,
                                    default_action=Vec2(0.1, 0.1),
                                    rng=rng3
                                   )
        end,

        "pomcpow" => begin
            rng3 = copy(p_rng)
            action_gen = AdaptiveRadiusRandom(max_radius=6.0, to_zero_first=true, to_light_second=true, rng=rng3)
            tree_queries = 1_000_000
            solver = POMCPOWSolver(next_action=action_gen,
                                    tree_queries=tree_queries,
                                    criterion=MaxUCB(5.0),
                                    final_criterion=MaxTries(),
                                    max_depth=40,
                                    k_action=10.0,
                                    alpha_action=1/8,
                                    k_observation=4.0,
                                    alpha_observation=1/8,
                                    estimate_value=OneStepValue(),
                                    rng=rng3
                                   )
        end,
        =#

        "pomcpow_10k" => begin
            rng3 = copy(p_rng)
            action_gen = AdaptiveRadiusRandom(max_radius=6.0, to_zero_first=true, to_light_second=true, rng=rng3)
            tree_queries = 10_000
            solver = POMCPOWSolver(next_action=action_gen,
                                    tree_queries=tree_queries,
                                    criterion=MaxUCB(5.0),
                                    final_criterion=MaxTries(),
                                    max_depth=40,
                                    k_action=10.0,
                                    alpha_action=1/8,
                                    k_observation=4.0,
                                    alpha_observation=1/8,
                                    estimate_value=OneStepValue(),
                                    rng=rng3
                                   )
        end,

        "greedy" => SimpleFeedback(gain=1.0, max_radius=10.0)

    )

    updaters = Dict{String, Any}(
        "heuristic" => :heuristic
    )
end

solver_keys = keys(solvers)
rewards = Dict{String, AbstractVector{Float64}}()
# state_hists = Dict{String, AbstractVector{AbstractVector{state_type(pomdp)}}}()
counts = Dict{String, AbstractVector{Int}}()
times = Dict{String, AbstractVector{Float64}}()
steps = Dict{String, AbstractVector{Int}}()

for (j, sk) in enumerate(solver_keys)
    s_rewards = SharedArray(Float64, N) 
    s_counts = SharedArray(Int, N)
    # s_hists = SharedArray(AbstractVector{state_type(pomdp)}, N)
    s_times = SharedArray(Float64, N)
    s_steps = SharedArray(Float64, N)
    prog = Progress(N, desc="$sk ($j of $(length(solver_keys)))...")
    pmap(prog, 1:N) do i 
        sim_rng = MersenneTwister(i)
        up_rng = MersenneTwister(i+100_000)
        if isa(solvers[sk], Policy)
            policy = deepcopy(solvers[sk])
        else
            policy = solve(solvers[sk], pomdp)
        end
        if get(updaters, sk, :standard) == :standard
            up = ObsAdaptiveParticleFilter(deepcopy(pomdp), LowVarianceResampler(100_000), 0.05, up_rng)
        elseif updaters[sk] == :heuristic
            up = updater(policy)
        end
        sim = HistoryRecorder(max_steps=40, rng=sim_rng)
        pomdp.count = 0
        s_times[i] = @CPUelapsed hist = simulate(sim, deepcopy(pomdp), policy, up)
        s_rewards[i] = discounted_reward(hist)
        s_counts[i] = pomdp.count
        s_steps[i] = n_steps(hist)
    end
    rewards[sk] = sdata(s_rewards)
    @show mean(rewards[sk])
    # state_hists[sk] = sdata(s_hists)
    counts[sk] = sdata(s_counts)
    times[sk] = sdata(s_times)
    steps[sk] = sdata(s_steps)
    @show sum(counts[sk])/sum(steps[sk])
end

for k in solver_keys
    println("$k mean: $(mean(rewards[k])) sem: $(std(rewards[k])/sqrt(N))")
    println("$k time: $(mean(times[k])) sim counts: $(mean(counts[k])) steps: $(mean(steps[k]))")
end

filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "data", "compare_$(Dates.format(now(), "E_d_u_HH_MM")).jld")
println("saving to $filename...")
@save(filename, pomdp, solver_keys, solvers, rewards, updaters, counts, steps)
println("done.")
