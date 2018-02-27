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
using CSV

file_contents = readstring(@__FILE__())

pomdp = SubHuntPOMDP()

vs = ValueIterationSolver()
if !isdefined(:vp) || vp.mdp != pomdp
    vp = solve(vs, pomdp, verbose=true)
end
qs = QMDPSolver()
qp = QMDP.create_policy(qs, pomdp)
qp.alphas[:] = vp.qmat

@show max_depth = 20
@show max_time = 1.0
@show N = 1000

alldata = DataFrame()
@show max_time
solvers = Dict{String, Union{Solver,Policy}}(
    "qmdp" => qp,
    "ping_first" => PingFirst(qp),

    "d_despot" => begin
        rng = MersenneTwister(13)
        b = IndependentBounds(DefaultPolicyLB(qp), 100.0, check_terminal=true)
        sol = DESPOTSolver(lambda=0.01,
                     epsilon_0=0.0,
                     K=500,
                     D=max_depth,
                     max_trials=1_000_000,
                     T_max=max_time,
                     bounds=b,
                     default_action=ReportWhenUsed(1),
                     rng=rng)
    end,

    "d_pomcp" => begin
        rng = MersenneTwister(13)
        sol = POMCPSolver(max_depth=max_depth,
                    max_time=max_time,
                    c=100.0,
                    tree_queries=typemax(Int),
                    default_action=ReportWhenUsed(qp),
                    estimate_value=FOValue(vp),
                    tree_in_info=false,
                    rng=rng
                   )
    end,

    "pomcpow" => begin
        rng = MersenneTwister(13)
        solver = POMCPOWSolver(tree_queries=10_000_000_000,
                               criterion=MaxUCB(17.0),
                               final_criterion=MaxTries(),
                               max_depth=max_depth,
                               max_time=max_time,
                               enable_action_pw=false,
                               k_observation=6.0,
                               alpha_observation=1/100.0,
                               estimate_value=FOValue(vp),
                               check_repeat_obs=false,
                               default_action=ReportWhenUsed(qp),
                               tree_in_info=false,
                               rng=rng
                              )
    end
)



for binsize in logspace(-2, 1, 7)
    for (k, solver) in solvers
    # test = ["qmdp", "pomcpow", "pft"]
    # for (k, solver) in [(s, solvers[s]) for s in test]
        dpomdp = DSubHuntPOMDP(pomdp, binsize)
        @show k, binsize
        if isa(solver, Solver)
            planner = solve(solver, dpomdp)
        else
            planner = solver
        end
        sims = []
        for i in 1:N
            srand(planner, i+50_000)
            filter = SIRParticleFilter(deepcopy(pomdp), 100_000, rng=MersenneTwister(i+90_000))            

            md = Dict(:solver=>k, :i=>i, :max_time=>max_time, :binsize=>binsize)
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
        if isempty(alldata)
            alldata = data
        else
            alldata = vcat(alldata, data)
        end
    end
end


datestring = Dates.format(now(), "E_d_u_HH_MM")
copyname = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "icaps_2018", "data", "subhunt_discretization_$(datestring).jl")
write(copyname, file_contents)
filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "icaps_2018", "data", "subhunt_discretization_$(datestring).csv")
println("saving to $filename...")
CSV.write(filename, alldata)
println("done.")
