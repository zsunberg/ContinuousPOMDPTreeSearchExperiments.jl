function MCTS.estimate_value(est::BasicPOMCP.SolvedFORollout,
                             bmdp::POMDPToolbox.GenerativeBeliefMDP, 
                             belief,
                             d::Int64)
    sim = RolloutSimulator(est.rng, Nullable{Any}(), Nullable{Float64}(), Nullable(d))
    return simulate(sim, bmdp.pomdp, est.policy, rand(est.rng, belief))
end

Base.srand(planner::Policy, x) = planner

struct GBMDPSolver <: Solver
    mdp_solver::Solver
    updater::Union{Function, Updater}
end

function solve(solver::GBMDPSolver, pomdp::POMDP)
    if isa(solver.updater, Function)
        updater = solver.updater(pomdp)
    else
        updater = solver.updater
    end
    belief_mdp = GenerativeBeliefMDP(deepcopy(pomdp), updater)
    return solve(solver.mdp_solver, belief_mdp)
end

function solve(qs::QMDPSolver, bmdp::GenerativeBeliefMDP)
    return solve(qs, bmdp.pomdp)
end
