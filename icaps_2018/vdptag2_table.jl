using ContinuousPOMDPTreeSearchExperiments
using ParticleFilters
using ARDESPOT
using BasicPOMCP
using POMCPOW
using POMDPs
using DiscreteValueIteration
using QMDP
using MCTS
using VDPTag2
using POMDPToolbox
using DataFrames

file_contents = readstring(@__FILE__())

pomdp = VDPTagPOMDP()
dpomdp = AODiscreteVDPTagPOMDP(pomdp, 30, 0.5)

@show max_time = 1.0
@show max_depth = 10

solvers = Dict{String, Union{Solver,Policy}}(
    "to_next" => ToNextML(mdp(pomdp)),
    "manage_uncertainty" => ManageUncertainty(pomdp, 0.01),

    "pomcpow" => begin
        rng = MersenneTwister(13)
        ro = ToNextMLSolver(rng)
        solver = POMCPOWSolver(tree_queries=10_000_000,
                               criterion=MaxUCB(65.0),
                               final_criterion=MaxTries(),
                               max_depth=max_depth,
                               max_time=max_time,
                               k_action=12.0,
                               alpha_action=1/8,
                               k_observation=1.0,
                               alpha_observation=1/30,
                               estimate_value=FORollout(ro),
                               next_action=RootToNextMLFirst(rng),
                               check_repeat_obs=false,
                               check_repeat_act=false,
                               default_action=ReportWhenUsed(TagAction(false, 0.0)),
                               rng=rng
                              )
    end,

    "pft" => begin
        rng = MersenneTwister(13)
        m = 10
        node_updater = ObsAdaptiveParticleFilter(deepcopy(pomdp),
                                           LowVarianceResampler(m),
                                           0.05, rng)            
        ro = ToNextML(mdp(pomdp), rng)
        solver = DPWSolver(n_iterations=typemax(Int),
                           exploration_constant=65.0,
                           depth=max_depth,
                           max_time=max_time,
                           k_action = 12.0, 
                           alpha_action = 1/8,
                           k_state = 4.0,
                           alpha_state = 1/20,
                           check_repeat_state=false,
                           check_repeat_action=false,
                           estimate_value=RolloutEstimator(ro),
                           next_action=RootToNextMLFirst(rng),
                           rng=rng
                          )
        belief_mdp = GenerativeBeliefMDP(deepcopy(pomdp), node_updater)
        solve(solver, belief_mdp)
    end,

    "pomcpdpw" => begin
        rng = MersenneTwister(13)
        ro = ToNextMLSolver(rng)
        sol = PDPWSolver(max_depth=max_depth,
                    max_time=max_time,
                    c=65.0,
                    k_action=12.0,
                    alpha_action=1/8,
                    k_observation=4.0,
                    alpha_observation=1/20,
                    enable_action_pw=true,
                    check_repeat_obs=false,
                    check_repeat_act=false,
                    tree_queries=typemax(Int),
                    # default_action=ReportWhenUsed(1),
                    estimate_value=FORollout(ro),
                    next_action=RootToNextMLFirst(rng),
                    rng=rng
                   )
        planner = solve(sol, dpomdp)
        translate_policy(planner, dpomdp, pomdp, dpomdp)
    end,

    "d_despot" => begin
        rng = MersenneTwister(13)
        ro = ToNextMLSolver(rng)
        b = IndependentBounds(DefaultPolicyLB(ro), VDPUpper())
        sol = DESPOTSolver(lambda=0.01,
                     K=100,
                     D=max_depth,
                     max_trials=1_000_000,
                     T_max=max_time,
                     bounds=b,
                     random_source=MemorizingSource(500, 10, rng, min_reserve=8),
                     rng=rng)
        planner = solve(sol, dpomdp)
        translate_policy(planner, dpomdp, pomdp, dpomdp)
    end, 

    "d_pomcp" => begin
        rng = MersenneTwister(13)
        ro = ToNextMLSolver(rng)
        sol = POMCPSolver(max_depth=max_depth,
                    max_time=max_time,
                    c=65.0,
                    tree_queries=typemax(Int),
                    default_action=ReportWhenUsed(1),
                    estimate_value=FORollout(ro),
                    rng=rng
                   )
        planner = solve(sol, dpomdp)
        translate_policy(planner, dpomdp, pomdp, dpomdp)
    end
)

@show N=1000

alldata = DataFrame()
for (k, solver) in solvers
# test = ["pomcpdpw"]
# for (k, solver) in [(s, solvers[s]) for s in test]
    @show k
    if isa(solver, Solver)
        planner = solve(solver, pomdp)
    else
        planner = solver
    end
    sims = []
    for i in 1:N
        srand(planner, i+50_000)
        filter = SIRParticleFilter(deepcopy(pomdp), 100_000, rng=MersenneTwister(i+90_000))            

        md = Dict(:solver=>k, :i=>i)
        sim = Sim(deepcopy(pomdp),
                  planner,
                  filter,
                  rng=MersenneTwister(i+70_000),
                  max_steps=100,
                  metadata=md
                 )

        push!(sims, sim)
    end

    data = run_parallel(sims)
    # data = run(sims)

    rs = data[:reward]
    println(@sprintf("reward: %6.3f ± %6.3f", mean(rs), std(rs)/sqrt(length(rs))))
end

#=
datestring = Dates.format(now(), "E_d_u_HH_MM")
copyname = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "icaps_2018", "data", "subhunt_table_$(datestring).jl")
write(copyname, file_contents)
filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "icaps_2018", "data", "subhunt_$(datestring).csv")
println("saving to $filename...")
writetable(filename, alldata)
println("done.")
=#
