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
                     K=1000,
                     D=max_depth,
                     max_trials=1_000_000,
                     T_max=max_time,
                     bounds=b,
                     # default_action=qp,
                     rng=rng)
    end,
)

@show N=500

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

        sim = Sim(deepcopy(pomdp),
                  planner,
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

#=
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
    b = IndependentBounds(DefaultPolicyLB(ro), 10.0)
    DESPOTSolver(lambda=0.01,
                 epsilon_0=0.0,
                 K=500,
                 D=max_depth,
                 max_trials=1_000_000,
                 T_max=max_time,
                 bounds=b,
                 default_action=ro,
                 rng=rng)
end,

"despot_01" => begin
    rng = MersenneTwister(13)
    ro = QMDPSolver()
    b = IndependentBounds(DefaultPolicyLB(ro), 10.0)
    DESPOTSolver(lambda=0.01,
                 epsilon_0=0.0,
                 K=500,
                 D=max_depth,
                 max_trials=1_000_000,
                 T_max=max_time,
                 bounds=b,
                 default_action=ro,
                 rng=rng)
end,
=#
