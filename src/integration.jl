function MCTS.estimate_value(est::BasicPOMCP.SolvedFORollout,
                             bmdp::POMDPToolbox.GenerativeBeliefMDP, 
                             belief,
                             d::Int64)
    sim = RolloutSimulator(est.rng, Nullable{Any}(), Nullable{Float64}(), Nullable(d))
    return simulate(sim, bmdp.pomdp, est.policy, rand(est.rng, belief))
end

Base.srand(planner::Policy, x) = planner
