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

N = 100

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
        "heuristic" => begin
            homing_policy = SimpleFeedback(gain=1.0, max_radius=10.0)
            exploring_up = ObsAdaptiveParticleFilter(pomdp, LowVarianceResampler(1000), 0.05, p_rng)
            homing_up = ObsAdaptiveParticleFilter(pomdp, LowVarianceResampler(1000), 0.0, p_rng)
            InfoGatherHeur(5.0, 0.03, homing_policy, exploring_up, homing_up)
        end,

        "vanilla_pomcp" => begin
            rng3 = copy(p_rng)
            action_gen = AdaptiveRadiusRandom(max_radius=6.0, to_zero_first=true, to_light_second=true, rng=rng3)
            rollout_policy = RadiusRandom(radius=10.0, rng=rng3)
            tree_queries = 50_000
            solver = POMCPDPWSolver(next_action=action_gen,
                                    tree_queries=tree_queries,
                                    c=5.0,
                                    max_depth=40,
                                    k_action=10.0,
                                    alpha_action=1/8,
                                    k_observation=4.0,
                                    alpha_observation=1/8,
                                    estimate_value=RolloutEstimator(rollout_policy),
                                    default_action=(b, ex)->Vec2(0.1, 0.1),
                                    rng=rng3
                                   )
        end,

        "modified_pomcp" => begin
            rng3 = copy(p_rng)
            action_gen = AdaptiveRadiusRandom(max_radius=6.0, to_zero_first=true, to_light_second=true, rng=rng3)
            rollout_policy = SimpleFeedback(max_radius=10.0)
            tree_queries = 50_000
            node_updater = ObsAdaptiveParticleFilter(pomdp, LowVarianceResampler(100), 0.05, rng3)
            # node_updater = SimpleParticleFilter(pomdp, LowVarianceResampler(100), rng=rng3)
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
                                    default_action=(b, ex)->Vec2(0.1, 0.1),
                                    rng=rng3
                                   )
        end,

        "pomcpow" => begin
            rng3 = copy(p_rng)
            action_gen = AdaptiveRadiusRandom(max_radius=6.0, to_zero_first=true, to_light_second=true, rng=rng3)
            tree_queries = 1_000_000
            node_updater = ObsAdaptiveParticleFilter(pomdp, LowVarianceResampler(100), 0.05, rng3)
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
                                    node_belief_updater=node_updater,
                                    rng=rng3
                                   )
        end,

        "greedy" => SimpleFeedback(gain=1.0, max_radius=10.0)
    )

    up_rng = MersenneTwister(5)
    standard_up = ObsAdaptiveParticleFilter(pomdp, LowVarianceResampler(100_000), 0.05, up_rng)
    updaters = Dict{String, Updater}(
        "vanilla_pomcp" => standard_up,
        "modified_pomcp" => standard_up,
        "greedy" => standard_up,
        # "pomcpow" => standard_up,
        "heuristic" => updater(solvers["heuristic"])
    )
end

solver_keys = keys(solvers)
rewards = Dict{String, AbstractVector{Float64}}()

for (j, sk) in enumerate(solver_keys)
    s_rewards = SharedArray(Float64, N) 
    prog = Progress(N, desc="$sk ($j of $(length(solver_keys)))...")
    pmap(prog, 1:N) do i 
        sim_rng = MersenneTwister(i)
        if isa(solvers[sk], Policy)
            policy = deepcopy(solvers[sk])
        else
            policy = solve(solvers[sk], pomdp)
        end
        up = deepcopy(updaters[sk])
        sim = RolloutSimulator(max_steps=40, rng=sim_rng)
        s_rewards[i] = simulate(sim, pomdp, policy, up)
    end
    rewards[sk] = sdata(s_rewards)
end

for k in solver_keys
    println("$k mean: $(mean(rewards[k])) sem: $(std(rewards[k])/sqrt(N))")
end

filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "data", "compare_$(Dates.format(now(), "E_d_u_HH_MM")).jld")
# @save(filename, pomdp, solver_keys, solvers, rewards, updaters)
