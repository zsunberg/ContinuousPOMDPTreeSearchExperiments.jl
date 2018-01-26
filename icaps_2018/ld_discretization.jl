using ContinuousPOMDPTreeSearchExperiments
using ParticleFilters
using ARDESPOT
using BasicPOMCP
using POMCPOW
using POMDPs
using QMDP
using MCTS
using DiscreteValueIteration
using POMDPToolbox
using ProgressMeter
using PmapProgressMeter

@everywhere using ContinuousPOMDPTreeSearchExperiments
@everywhere using POMDPs
@everywhere using ParticleFilters

@show max_time = 1.0
@show max_depth = 20
pomdp = SimpleLightDark()

solvers = Dict{String, Union{Solver,Policy}}(

    "pomcpow" => begin
        rng = MersenneTwister(13)
        ro = ValueIterationSolver()
        solver = POMCPOWSolver(tree_queries=10_000_000,
                               criterion=MaxUCB(90.0),
                               final_criterion=MaxTries(),
                               max_depth=max_depth,
                               max_time=max_time,
                               enable_action_pw=false,
                               k_observation=5.0,
                               alpha_observation=1/15.0,
                               estimate_value=FOValue(ro),
                               check_repeat_obs=false,
                               # default_action=ReportWhenUsed(-1),
                               rng=rng
                              )
    end,


    "d_pomcp" => begin
        rng = MersenneTwister(13)
        ro = ValueIterationSolver()
        sol = POMCPSolver(max_depth=max_depth,
                    max_time=max_time,
                    c=100.0,
                    tree_queries=typemax(Int),
                    # default_action=ro,
                    estimate_value=FORollout(ro),
                    rng=rng
                   )
    end,

    "d_despot" => begin
        rng = MersenneTwister(13)
        ro = QMDPSolver()
        b = IndependentBounds(-100.0, 100.0, check_terminal=true)
        sol = DESPOTSolver(lambda=0.01,
                     epsilon_0=0.0,
                     K=5000,
                     D=max_depth,
                     max_trials=1_000_000,
                     T_max=max_time,
                     bounds=b,
                     default_action=ReportWhenUsed(solve(ro, pomdp)),
                     rng=rng)
    end,

    "qmdp" => QMDPSolver(),
)

@show N=1000

# for (k, solver) in solvers
test = keys(solvers)
for d in logspace(-2, 1, 7)
    dpomdp = DSimpleLightDark(pomdp, d)
    for (k, solver) in [(k, solvers[k]) for k in test]
        @show k, d
        if isa(solver, Solver)
            planner = solve(solver, dpomdp)
        else
            planner = solver
        end
        # prog = Progress(N, desc="Creating Simulations...")
        # sims = pmap(prog, 1:N) do i
        sims = []
        for i in 1:N
            srand(planner, i+50_000)
            filter = ObsAdaptiveParticleFilter(deepcopy(pomdp),
                                               LowVarianceResampler(10_000),
                                               0.05, MersenneTwister(i+90_000))            

            sim = Sim(deepcopy(pomdp),
                planner,
                filter,
                rng=MersenneTwister(i+70_000),
                max_steps=100,
                metadata=Dict(:d=>d, :solver=>k, :i=>i)
               )

            push!(sims, sim)
        end

        data = run_parallel(sims)
        # data = run(sims)

        rs = data[:reward]
        println(@sprintf("reward: %6.3f ± %6.3f", mean(rs), std(rs)/sqrt(length(rs))))
    end
end
