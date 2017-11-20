using ContinuousPOMDPTreeSearchExperiments
using ParticleFilters
using ARDESPOT
using BasicPOMCP
using POMCPOW
using POMDPs
using QMDP
using DiscreteValueIteration

@show max_time = 1.0
@show max_depth = 20
pomdp = SimpleLightDark()

solvers = Dict{String, Union{Solver,Policy}}(

    "pomcpow" => begin
        rng = MersenneTwister(13)
        ro = ValueIterationSolver()
        solver = POMCPOWSolver(tree_queries=10_000_000,
                               criterion=MaxUCB(40.0),
                               final_criterion=MaxTries(),
                               max_depth=max_depth,
                               max_time=max_time,
                               enable_action_pw=false,
                               k_observation=4.0,
                               alpha_observation=1/20,
                               estimate_value=FORollout(ro),
                               check_repeat_obs=false,
                               # default_action=ReportWhenUsed(-1),
                               rng=rng
                              )
    end,

    "pomcp" => begin
        rng = MersenneTwister(13)
        ro = ValueIterationSolver()
        POMCPSolver(max_depth=max_depth,
                    max_time=max_time,
                    c=40.0,
                    tree_queries=typemax(Int),
                    # default_action=ro,
                    estimate_value=FORollout(ro),
                    rng=rng
                   )
    end,

    "despot_0" => begin
        rng = MersenneTwister(13)
        ro = QMDPSolver()
        # b = IndependentBounds(DefaultPolicyLB(ro), FullyObservableValueUB(ro))
        b = IndependentBounds(DefaultPolicyLB(ro), 10.0, check_terminal=true)
        DESPOTSolver(lambda=0.01,
                     epsilon_0=0.0,
                     K=500,
                     D=max_depth,
                     max_trials=1_000_000,
                     T_max=max_time,
                     bounds=b,
                     default_action=ReportWhenUsed(solve(ro, pomdp)),
                     rng=rng)
    end,

    "despot_01" => begin
        rng = MersenneTwister(13)
        ro = QMDPSolver()
        b = IndependentBounds(DefaultPolicyLB(ro), 10.0, check_terminal=true)
        DESPOTSolver(lambda=0.01,
                     epsilon_0=0.0,
                     K=500,
                     D=max_depth,
                     max_trials=1_000_000,
                     T_max=max_time,
                     bounds=b,
                     default_action=ReportWhenUsed(solve(ro, pomdp)),
                     rng=rng)
    end,


    "qmdp" => QMDPSolver(),
    "heuristic_1" => LDHSolver(std_thresh=0.1),
    "heuristic_01" => LDHSolver(std_thresh=0.1)
)

@show N=20

for (k, solver) in solvers
    @show k
    if isa(solver, Solver)
        planner = solve(solver, pomdp)
    else
        planner = solver
    end
    sims = []
    for i in 1:N
        srand(planner, i+50_000)
        filter = ObsAdaptiveParticleFilter(deepcopy(pomdp),
                                           LowVarianceResampler(10_000),
                                           0.05, MersenneTwister(i+90_000))            

        sim = Sim(deepcopy(pomdp),
                  deepcopy(planner),
                  filter,
                  rng=MersenneTwister(i+70_000),
                  max_steps=100,
                  metadata=Dict(:solver=>k, :i=>i)
                 )

        push!(sims, sim)
    end

    data = run_parallel(sims)
    # data = run(sims)

    rs = data[:reward]
    println(@sprintf("reward: %6.3f ± %6.3f", mean(rs), std(rs)/sqrt(length(rs))))
end
