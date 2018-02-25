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
using LaserTag
using DataFrames
using CSV
using ProgressMeter
using PmapProgressMeter

@everywhere using LaserTag
@everywhere using ContinuousPOMDPTreeSearchExperiments
@everywhere using POMDPs
@everywhere using ParticleFilters

file_contents = readstring(@__FILE__())

@show max_time = 1.0
@show max_depth = 90
@show exploration = 26.0

P = typeof(gen_lasertag(rng=MersenneTwister(18), robot_position_known=false))

solvers = Dict{String, Union{Solver,Policy}}(
    "qmdp" => QMDPSolver(),

    #=
    Mon 04 Dec 2017 05:11:41 PM PST

    mean(combined[:mean_reward]) = -9.947048511007113
    iteration 88
    mean(d) = [26.3468, 3.62511, 35.0585]
    ev = eigvals(cov(d)) = [2.93607e-11, 6.44598e-8, 0.00381188]
    (eigvecs(cov(d)))[:, j] = [0.00247583, -0.998464, -0.0553574]
    (eigvecs(cov(d)))[:, j] = [0.381521, -0.0502272, 0.922995]
    (eigvecs(cov(d)))[:, j] = [-0.924357, -0.0234052, 0.38081]
    =#

    "pomcpow" => begin
        rng = MersenneTwister(13)
        solver = POMCPOWSolver(tree_queries=10_000_000,
                               criterion=MaxUCB(exploration),
                               final_criterion=MaxTries(),
                               max_depth=max_depth,
                               max_time=max_time,
                               enable_action_pw=false,
                               k_observation=4.0,
                               alpha_observation=1/35,
                               estimate_value=FOValue(ValueIterationSolver()),
                               check_repeat_obs=false,
                               default_action=LaserTag.TAG_ACTION,
                               rng=rng
                              )
    end,

    "pft" => begin
        rng = MersenneTwister(13)
        m = 20
        solver = DPWSolver(n_iterations=typemax(Int),
                           exploration_constant=exploration,
                           depth=max_depth,
                           max_time=max_time,
                           k_state=4.0,
                           alpha_state=1/35,
                           check_repeat_state=false,
                           check_repeat_action=false,
                           estimate_value=RolloutEstimator(QMDPSolver()),
                           enable_action_pw=false,
                           # default_action=ReportWhenUsed(NoGapTag()),
                           rng=rng
                          )
        GBMDPSolver(solver, pomdp->ObsAdaptiveParticleFilter(pomdp,
                                                             LowVarianceResampler(m),
                                                             0.1, rng))
    end,

    "despot" => begin
        rng = MersenneTwister(13)
        # b = IndependentBounds(DefaultPolicyLB(QMDPSolver()), 100.0, check_terminal=true)
        bounds = LaserBounds{P}()
        K = 500
        DESPOTSolver(lambda=0.01,
                     K=K,
                     D=max_depth,
                     max_trials=1_000_000,
                     T_max=max_time,
                     bounds=bounds,
                     bounds_warnings=false,
                     default_action=LaserTag.TAG_ACTION,
                     random_source=MemorizingSource(K, max_depth, rng, min_reserve=10),
                     rng=rng)
    end,

    "pomcp" => begin
        rng = MersenneTwister(13)
        POMCPSolver(max_depth=max_depth,
                    max_time=max_time,
                    c=exploration,
                    tree_queries=typemax(Int),
                    default_action=LaserTag.TAG_ACTION,
                    estimate_value=FOValue(ValueIterationSolver()),
                    rng=rng
                   )
    end,

    "pomcpdpw" => begin
        rng = MersenneTwister(13)
        solver = PDPWSolver(tree_queries=10_000_000,
                            c=exploration,
                            max_depth=max_depth,
                            max_time=max_time,
                            enable_action_pw=false,
                            k_observation=4.0,
                            alpha_observation=1/35.0,
                            estimate_value=FOValue(ValueIterationSolver()),
                            check_repeat_obs=false,
                            # default_action=ReportWhenUsed(-1),
                            rng=rng
                           )
    end,
)

@show N=1000

alldata = DataFrame()
for (k, solver) in solvers
# test = ["qmdp", "pomcpow", "pft"]
# for (k, solver) in [(s, solvers[s]) for s in test]
# k = "qmdp"
# solver = QMDPSolver()
    @show k
    prog = Progress(N, desc="Creating Simulations...")
    sims = pmap(prog, 1:N) do i
        pomdp = gen_lasertag(rng=MersenneTwister(i+300_000))
        planner = solve(solver, pomdp)
        srand(planner, i+50_000)
        up_rng = MersenneTwister(i+140_000)
        filter = ObsAdaptiveParticleFilter(pomdp, LowVarianceResampler(100_000), 0.05, up_rng)

        md = Dict(:solver=>k, :i=>i)
        return Sim(pomdp,
            planner,
            filter,
            rng=MersenneTwister(i+70_000),
            max_steps=100,
            metadata=md
           )
    end

    data = run_parallel(sims)
    # data = run(sims)


    rs = data[:reward]
    println(@sprintf("reward: %6.3f ± %6.3f", mean(rs), std(rs)/sqrt(length(rs))))
    if isempty(alldata)
        alldata = data
    else
        alldata = vcat(alldata, data)
    end
end

datestring = Dates.format(now(), "E_d_u_HH_MM")
copyname = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "icaps_2018", "data", "lasertag_table_$(datestring).jl")
write(copyname, file_contents)
filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "icaps_2018", "data", "lasertag_$(datestring).csv")
println("saving to $filename...")
CSV.write(filename, alldata)
println("done.")
