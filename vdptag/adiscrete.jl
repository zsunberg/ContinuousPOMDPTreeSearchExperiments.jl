using POMDPs
using ContinuousPOMDPTreeSearchExperiments
using POMDPToolbox
using BasicPOMCP
using POMCPOW
using ParticleFilters
using CPUTime
using VDPTag
using DataFrames
using ARDESPOT

N = 100

pomdp = VDPTagPOMDP()
@show mdp(pomdp).tag_radius
@show mdp(pomdp).tag_terminate

file_contents = readstring(@__FILE__())

solvers = Dict{String, Union{Solver,Policy}}(

    "pomcpow" => begin
        rng = MersenneTwister(13)
        ro = ToNextMLSolver(rng)
        solver = POMCPOWSolver(tree_queries=10_000_000,
                               criterion=MaxUCB(40.0),
                               final_criterion=MaxTries(),
                               max_depth=10,
                               max_time=0.1,
                               k_action=8.0,
                               alpha_action=1/20,
                               k_observation=4.0,
                               alpha_observation=1/20,
                               estimate_value=FORollout(ro),
                               check_repeat_act=true,
                               check_repeat_obs=true,
                               next_action=RootToNextMLFirst(rng),
                               # default_action=1,
                               rng=rng
                              )
    end,

    "pomcp" => begin
        rng = MersenneTwister(13)
        ro = ToNextMLSolver(rng)
        POMCPSolver(max_depth=10,
                    max_time=0.1,
                    c=40.0,
                    tree_queries=typemax(Int),
                    # default_action=1,
                    estimate_value=FORollout(ro),
                    rng=rng
                   )
    end,

    "despot" => begin
        rng = MersenneTwister(13)
        ro = ToNextMLSolver(rng)
        b = IndependentBounds(DefaultPolicyLB(ro), VDPUpper())
        DESPOTSolver(lambda=0.01,
                     K=500,
                     D=10,
                     max_trials=1_000_000,
                     T_max=0.1,
                     bounds=b,
                     random_source=MemorizingSource(500, 10, rng, min_reserve=8),
                     rng=rng)
    end
)

alldata = DataFrame()
for n_angles_float in logspace(0.5, 3, 6)
# for n_obs_angles_float in logspace(0.5, 3, 6)
# for n_angles_float in [10.0]
    n_angles = round(Int, n_angles_float)
    for (k, solver) in solvers
        println("$k ($n_angles)")
        dpomdp = ADiscreteVDPTagPOMDP(n_angles=n_angles)
        planner = solve(solver, dpomdp)
        sims = []
        for i in 1:N
            srand(planner, i+50_000)
            cplanner = translate_policy(planner, dpomdp, pomdp, dpomdp)

            filter = ObsAdaptiveParticleFilter(deepcopy(pomdp),
                                               LowVarianceResampler(10_000),
                                               0.05, MersenneTwister(i+90_000))            

            sim = Sim(deepcopy(pomdp),
                      deepcopy(cplanner),
                      filter,
                      rng=MersenneTwister(i+70_000),
                      max_steps=100,
                      metadata=Dict(:solver=>k,
                                    :n_angles=>n_angles,
                                    :i=>i)
                     )

            push!(sims, sim)
        end

        data = run_parallel(sims)
        # data = run(sims)

        rs = data[:reward]
        println(@sprintf("reward: %6.3f ± %6.3f", mean(rs), std(rs)/sqrt(length(rs))))
        alldata = vcat(alldata, data)
    end
end
# end

filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "data", "vdp_a_discretization_$(Dates.format(now(), "E_d_u_HH_MM"))")
println("saving to $filename...")
writetable(filename*".csv", alldata)
write(filename*"_file.jl", file_contents)
println("done.")
