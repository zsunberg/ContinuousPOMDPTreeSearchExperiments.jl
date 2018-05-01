using ContinuousPOMDPTreeSearchExperiments
using Multilane
using POMCPOW
using POMDPs
using MCTS
using ARDESPOT
using DataFrames
using CSV

file_contents = readstring(@__FILE__())

@show cor = 0.0
@show lambda = 2.0

@show N = 1
@show n_iters = 1_000_000
@show max_time = 1.0
@show max_depth = 40
@show val = SimpleSolver()
alldata = DataFrame()

dpws = DPWSolver(depth=max_depth,
                 n_iterations=n_iters,
                 max_time=max_time,
                 exploration_constant=8.0,
                 k_state=4.5,
                 alpha_state=1/10.0,
                 enable_action_pw=false,
                 check_repeat_state=false,
                 estimate_value=RolloutEstimator(val)
                 # estimate_value=val
                )

solvers = Dict{String, Solver}(
    "qmdp" => QBSolver(dpws),
    "pftdpw_5" => begin
        m = 5
        wup = WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.05)
        rng = MersenneTwister(123)
        up = BehaviorParticleUpdater(nothing, m, 0.0, 0.0, wup, rng)
        BBMDPSolver(dpws, up)
    end,
    "pomcpow" => begin
        wup = WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.05) 
        POMCPOWSolver(tree_queries=n_iters,
                               criterion=MaxUCB(8.0),
                               max_depth=max_depth,
                               max_time=max_time,
                               enable_action_pw=false,
                               k_observation=4.5,
                               alpha_observation=1/10.0,
                               estimate_value=FORollout(val),
                               # estimate_value=val,
                               check_repeat_obs=false,
                               node_sr_belief_updater=BehaviorPOWFilter(wup)
                              )
    end,
    "despot" => begin
        rng = MersenneTwister(13)
        b = IndependentBounds(DefaultPolicyLB(val), 1.02, check_terminal=true)
        sol = DESPOTSolver(lambda=0.01,
                     K=100,
                     D=max_depth,
                     max_trials=1_000_000,
                     T_max=max_time,
                     bounds=b,
                     random_source=MemorizingSource(500, max_depth, rng, min_reserve=25),
                     default_action=ReportWhenUsed(MLAction(0.0, 0.0)),
                     rng=rng)
    end, 
    "simple" => SimpleSolver()
)

behaviors = standard_uniform(correlation=cor)
pp = PhysicalParam(4, lane_length=100.0)
dmodel = NoCrashIDMMOBILModel(10, pp,
                              behaviors=behaviors,
                              p_appear=1.0,
                              lane_terminate=true,
                              max_dist=1000.0
                             )
rmodel = SuccessReward(lambda=lambda)
pomdp = NoCrashPOMDP{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 0.95, false)
mdp = NoCrashMDP{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 0.95, false)

for (k, solver) in solvers
    @show k

    sims = []
    for i in 1:N
        if k == "qmdp"
            planner = deepcopy(solve(solver, mdp))
        else
            planner = deepcopy(solve(solver, pomdp))
        end
        srand(planner, i+50_000)
        wup = WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.05) 
        filter = BehaviorParticleUpdater(pomdp, 5000, 0.0, 0.0, wup, MersenneTwister(i+50_000))
        # filter = AggressivenessUpdater(pomdp, 5000, 0.0, 0.0, wup, MersenneTwister(i+50_000))

        rng = MersenneTwister(i+70_000)
        is = initial_state(pomdp, rng)
        ips = MLPhysicalState(is)

        md = Dict(:solver=>k, :i=>i)
        sim = Sim(deepcopy(pomdp),
                  planner,
                  filter,
                  ips,
                  is,
                  rng=rng,
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
copyname = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "data", "multilane_table_$(datestring).jl")
write(copyname, file_contents)
filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "data", "multilane_$(datestring).csv")
println("saving to $filename...")
CSV.write(filename, alldata)
println("done.")
