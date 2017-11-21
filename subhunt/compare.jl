using ContinuousPOMDPTreeSearchExperiments
using ParticleFilters
using ARDESPOT
using BasicPOMCP
using POMCPOW
using POMDPs
using DiscreteValueIteration
using QMDP
using SubHunt

pomdp = SubHuntPOMDP()

vs = ValueIterationSolver()
vp = solve(vs, pomdp, verbose=true)
qs = QMDPSolver()
qp = QMDP.create_policy(qs, pomdp)
qp.alphas[:] = vp.qmat

@show max_time = 2.0
@show max_depth = 20

solvers = Dict{String, Union{Solver,Policy}}(
    "qmdp" => qp,
    "ping_first" => PingFirst(qp),

    #=
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
                               estimate_value=FOValue(vp),
                               check_repeat_obs=false,
                               # default_action=ReportWhenUsed(-1),
                               rng=rng
                              )
    end,

    "despot_01" => begin
        rng = MersenneTwister(13)
        b = IndependentBounds(DefaultPolicyLB(qp), 100.0, check_terminal=true)
        DESPOTSolver(lambda=0.01,
                     epsilon_0=0.0,
                     K=500,
                     D=max_depth,
                     max_trials=1_000_000,
                     T_max=max_time,
                     bounds=b,
                     # default_action=qp,
                     rng=rng)
    end,
    =#

    "pomcpdpw" => begin
        rng = MersenneTwister(13)
        solver = PDPWSolver(tree_queries=10_000_000,
                            c=100.0,
                            max_depth=max_depth,
                            max_time=max_time,
                            enable_action_pw=false,
                            k_observation=1.0,
                            alpha_observation=1/10,
                            estimate_value=FOValue(vp),
                            check_repeat_obs=false,
                            # default_action=ReportWhenUsed(-1),
                            rng=rng
                           )
    end,
)

#=
for d in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
    solvers["despot_$(d)_01"] = begin
        rng = MersenneTwister(13)
        b = IndependentBounds(DefaultPolicyLB(qp), 100.0, check_terminal=true)
        dpomdp = DSubHuntPOMDP(pomdp, d)
        sol = DESPOTSolver(lambda=0.01,
                     epsilon_0=0.0,
                     K=500,
                     D=max_depth,
                     max_trials=1_000_000,
                     T_max=max_time,
                     bounds=b,
                     default_action=ReportWhenUsed(qp),
                     rng=rng)
        solve(sol, dpomdp)
    end
end
=#

@show N=1

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
        filter = SIRParticleFilter(deepcopy(pomdp), 100_000, rng=MersenneTwister(i+90_000))            

        md = Dict(:solver=>k, :i=>i)
        if isa(planner, DESPOTPlanner{DSubHuntPOMDP})
            md[:d] = planner.pomdp.binsize
        end
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

    rs = data[:reward]
    println(@sprintf("reward: %6.3f ± %6.3f", mean(rs), std(rs)/sqrt(length(rs))))
end
