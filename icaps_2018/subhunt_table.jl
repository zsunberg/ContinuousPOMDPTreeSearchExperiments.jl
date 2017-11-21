using ContinuousPOMDPTreeSearchExperiments
using ParticleFilters
using ARDESPOT
using BasicPOMCP
using POMCPOW
using POMDPs
using DiscreteValueIteration
using QMDP
using MCTS
using SubHunt
using POMDPToolbox
using DataFrames

file_contents = readstring(@__FILE__())

pomdp = SubHuntPOMDP()

vs = ValueIterationSolver()
if !isdefined(:vp) || vp.mdp != pomdp
    vp = solve(vs, pomdp, verbose=true)
end
qs = QMDPSolver()
qp = QMDP.create_policy(qs, pomdp)
qp.alphas[:] = vp.qmat

@show max_time = 1.0
@show max_depth = 20

solvers = Dict{String, Union{Solver,Policy}}(
    "qmdp" => qp,
    "ping_first" => PingFirst(qp),

    "pomcpow" => begin
        rng = MersenneTwister(13)
        solver = POMCPOWSolver(tree_queries=10_000_000,
                               criterion=MaxUCB(100.0),
                               final_criterion=MaxTries(),
                               max_depth=max_depth,
                               max_time=max_time,
                               enable_action_pw=false,
                               k_observation=1.0,
                               alpha_observation=1/10,
                               estimate_value=FORollout(vp),
                               check_repeat_obs=false,
                               default_action=ReportWhenUsed(qp),
                               rng=rng
                              )
    end,

    "pft" => begin
        rng = MersenneTwister(13)
        m = 20
        node_updater = ObsAdaptiveParticleFilter(deepcopy(pomdp),
                                           LowVarianceResampler(m),
                                           0.1, rng)            
        solver = DPWSolver(n_iterations=typemax(Int),
                           exploration_constant=100.0,
                           depth=max_depth,
                           max_time=max_time,
                           k_state=2.0,
                           alpha_state=1/10,
                           check_repeat_state=false,
                           check_repeat_action=false,
                           estimate_value=RolloutEstimator(qp),
                           enable_action_pw=false,
                           # default_action=ReportWhenUsed(qp),
                           rng=rng
                          )
        belief_mdp = GenerativeBeliefMDP(deepcopy(pomdp), node_updater)
        solve(solver, belief_mdp)
    end,

    "ar_despot" => begin
        rng = MersenneTwister(13)
        b = IndependentBounds(DefaultPolicyLB(qp), 100.0, check_terminal=true)
        DESPOTSolver(lambda=0.01,
                     epsilon_0=0.0,
                     K=500,
                     D=max_depth,
                     max_trials=1_000_000,
                     T_max=max_time,
                     bounds=b,
                     default_action=ReportWhenUsed(1),
                     rng=rng)
    end,

    "d_despot" => begin
        rng = MersenneTwister(13)
        b = IndependentBounds(DefaultPolicyLB(qp), 100.0, check_terminal=true)
        dpomdp = DSubHuntPOMDP(pomdp, 1.0)
        sol = DESPOTSolver(lambda=0.01,
                     epsilon_0=0.0,
                     K=500,
                     D=max_depth,
                     max_trials=1_000_000,
                     T_max=max_time,
                     bounds=b,
                     default_action=ReportWhenUsed(1),
                     rng=rng)
        solve(sol, dpomdp)
    end,

    "pomcp" => begin
        rng = MersenneTwister(13)
        POMCPSolver(max_depth=max_depth,
                    max_time=max_time,
                    c=100.0,
                    tree_queries=typemax(Int),
                    default_action=ReportWhenUsed(qp),
                    estimate_value=FOValue(vp),
                    rng=rng
                   )
    end,

    "d_pomcp" => begin
        rng = MersenneTwister(13)
        dpomdp = DSubHuntPOMDP(pomdp, 1.0)
        sol = POMCPSolver(max_depth=max_depth,
                    max_time=max_time,
                    c=100.0,
                    tree_queries=typemax(Int),
                    default_action=ReportWhenUsed(qp),
                    estimate_value=FOValue(vp),
                    rng=rng
                   )
        solve(sol, dpomdp)
    end,
)

@show N=500

alldata = DataFrame()
# for (k, solver) in solvers
test = ["qmdp", "pomcpow", "pft"]
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
    alldata = vcat(alldata, data)
end

datestring = Dates.format(now(), "E_d_u_HH_MM")
copyname = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "icaps_2018", "data", "subhunt_table_$(datestring).jl")
write(copyname, file_contents)
filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "icaps_2018", "data", "subhunt_$(datestring).csv")
println("saving to $filename...")
writetable(filename, alldata)
println("done.")
