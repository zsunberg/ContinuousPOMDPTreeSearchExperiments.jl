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
using CSV


file_contents = readstring(@__FILE__())

pomdp = VDPTagPOMDP(mdp=VDPTagMDP(barriers=CardinalBarriers(0.2, 2.8)))
dpomdp = AODiscreteVDPTagPOMDP(pomdp, 25, 0.2)

@show max_time = 1.0
@show max_depth = 10
@show RO = RandomSolver
# @show RO = ToNextMLSolver

solvers = Dict{String, Union{Solver,Policy}}(
    "to_next" => ToNextML(mdp(pomdp)),
    "manage_uncertainty" => ManageUncertainty(pomdp, 0.01),

    "pomcpow" => begin
        rng = MersenneTwister(13)
        # ro = ToNextMLSolver(rng)::RO
        ro = RandomSolver(rng)::RO
        solver = POMCPOWSolver(tree_queries=10_000_000,
                               criterion=MaxUCB(110.0),
                               final_criterion=MaxQ(),
                               max_depth=max_depth,
                               max_time=max_time,
                               k_action=30.0,
                               alpha_action=1/30,
                               k_observation=5.0,
                               alpha_observation=1/100,
                               estimate_value=FORollout(ro),
                               next_action=RootToNextMLFirst(rng),
                               check_repeat_obs=false,
                               check_repeat_act=false,
                               tree_in_info=false,
                               default_action=ReportWhenUsed(TagAction(false, 0.0)),
                               rng=rng
                              )
    end,

    "pft" => begin
        rng = MersenneTwister(13)
        m = 15
        node_updater = ObsAdaptiveParticleFilter(deepcopy(pomdp),
                                           LowVarianceResampler(m),
                                           0.05, rng)            
        # ro = ToNextMLSolver(rng)::RO
        ro = RandomSolver(rng)::RO
        ev = SampleRollout(solve(ro, pomdp), rng)
        solver = DPWSolver(n_iterations=typemax(Int),
                           exploration_constant=85.0,
                           depth=max_depth,
                           max_time=max_time,
                           k_action = 20.0, 
                           alpha_action = 1/20,
                           k_state = 8.0,
                           alpha_state = 1/60,
                           check_repeat_state=false,
                           check_repeat_action=false,
                           tree_in_info=false,
                           estimate_value=ev,
                           next_action=RootToNextMLFirst(rng),
                           default_action=ReportWhenUsed(TagAction(false, 0.0)),
                           rng=rng
                          )
        belief_mdp = GenerativeBeliefMDP(deepcopy(pomdp), node_updater)
        solve(solver, belief_mdp)
    end,

    "mr_pft" => begin
        rng = MersenneTwister(13)
        m = 15

        ro = RandomSolver(rng)::RO
        # ev = RolloutEstimator(ro)
        ev = SampleRollout(solve(ro, pomdp), rng)
        solver = DPWSolver(n_iterations=typemax(Int),
                           exploration_constant=85.0,
                           depth=max_depth,
                           max_time=max_time,
                           k_action = 20.0, 
                           alpha_action = 1/20,
                           k_state = 8.0,
                           alpha_state = 1/60,
                           check_repeat_state=false,
                           check_repeat_action=false,
                           tree_in_info=false,
                           estimate_value=ev,
                           next_action=RootToNextMLFirst(rng),
                           # default_action=ReportWhenUsed(TagAction(false, 0.0)),
                           rng=rng
                          )
        belief_mdp = MeanRewardBeliefMDP(pomdp,
                                         LowVarianceResampler(m),
                                         0.05
                                        )
        solve(solver, belief_mdp)
    end,

    "pomcpdpw" => begin
        rng = MersenneTwister(13)
        # ro = ToNextMLSolver(rng)
        ro = RandomSolver(rng)::RO
        sol = PDPWSolver(max_depth=max_depth,
                    max_time=max_time,
                    c=110.0,
                    k_action=30.0,
                    alpha_action=1/30.0,
                    k_observation=5.0,
                    alpha_observation=1/100.0,
                    enable_action_pw=true,
                    check_repeat_obs=false,
                    check_repeat_act=false,
                    tree_queries=typemax(Int),
                    # default_action=ReportWhenUsed(1),
                    estimate_value=FORollout(ro),
                    next_action=RootToNextMLFirst(rng),
                    rng=rng
                   )
    end,

    "d_despot" => begin
        rng = MersenneTwister(13)
        # ro = ToNextMLSolver(rng)
        ro = RandomSolver(rng)::RO
        b = IndependentBounds(DefaultPolicyLB(ro), VDPUpper())
        sol = DESPOTSolver(lambda=0.01,
                     K=200,
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
        # ro = ToNextMLSolver(rng)
        ro = RandomSolver(rng)::RO
        sol = POMCPSolver(max_depth=max_depth,
                    max_time=max_time,
                    c=100.0,
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

# for (k, solver) in solvers
test = ["mr_pft"]
for (k, solver) in [(s, solvers[s]) for s in test]
    @show k
    if isa(solver, Solver)
        planner = solve(solver, pomdp)
    else
        planner = solver
    end
    sims = []
    for i in 1:N
        srand(planner, i+50_000)
        filter = SIRParticleFilter(deepcopy(pomdp), 10_000, rng=MersenneTwister(i+90_000))            

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

    # data = run_parallel(sims)
    data = run(sims)

    if isempty(alldata)
        alldata = data
    else
        alldata = vcat(alldata, data)
    end

    rs = data[:reward]
    println(@sprintf("reward: %6.3f ± %6.3f", mean(rs), std(rs)/sqrt(length(rs))))
end

datestring = Dates.format(now(), "E_d_u_HH_MM")
copyname = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "icaps_2018", "data", "vdpbarrier_table_$(datestring).jl")
write(copyname, file_contents)
filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "icaps_2018", "data", "vdpbarrier_$(datestring).csv")
println("saving to $filename...")
CSV.write(filename, alldata)
println("done.")
