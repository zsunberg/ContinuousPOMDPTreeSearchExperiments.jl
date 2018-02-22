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
using Plots
using StatPlots

@everywhere using POMDPToolbox

file_contents = readstring(@__FILE__())

pomdp = VDPTagPOMDP(mdp=VDPTagMDP(agent_speed=2.0#=, dt=0.01=#))
dpomdp = AODiscreteVDPTagPOMDP(pomdp, 30, 0.5)

@show max_time = 1.0
@show max_depth = 10

solvers = Dict{String, Union{Solver,Policy}}(
    "to_next" => ToNextML(mdp(pomdp)),
    "manage_uncertainty" => ManageUncertainty(pomdp, 0.01),

    "pomcpow" => begin
        rng = MersenneTwister(13)
        ro = ToNextMLSolver(rng)
        ro = RandomSolver(rng)
        solver = POMCPOWSolver(tree_queries=10_000_000,
                               criterion=MaxUCB(22.0),
                               final_criterion=MaxQ(),
                               max_depth=max_depth,
                               max_time=max_time,
                               k_action=35.0,
                               alpha_action=1/15.0,
                               k_observation=3.0,
                               alpha_observation=1/100.0,
                               estimate_value=FORollout(ro),
                               next_action=RootToNextMLFirst(rng),
                               check_repeat_obs=false,
                               check_repeat_act=false,
                               default_action=ReportWhenUsed(TagAction(false, 0.0)),
                               rng=rng
                              )
    end,

    "pft" => begin
        rng = MersenneTwister(13)
        m = 10
        node_updater = ObsAdaptiveParticleFilter(deepcopy(pomdp),
                                           LowVarianceResampler(m),
                                           0.05, rng)            
        ro = ToNextML(mdp(pomdp), rng)
        ro = RandomSolver(rng)
        solver = DPWSolver(n_iterations=typemax(Int),
                           exploration_constant=72.0,
                           depth=max_depth,
                           max_time=max_time,
                           k_action = 25.0, 
                           alpha_action = 1/17,
                           k_state = 28.0,
                           alpha_state = 1/1.6,
                           check_repeat_state=false,
                           check_repeat_action=false,
                           estimate_value=RolloutEstimator(ro),
                           next_action=RootToNextMLFirst(rng),
                           rng=rng
                          )
        belief_mdp = GenerativeBeliefMDP(deepcopy(pomdp), node_updater)
        solve(solver, belief_mdp)
    end,
)

@show N=100

alldata = DataFrame()

# for (k, solver) in solvers
test = ["pomcpow", "pft"]
# test = ["to_next", "manage_uncertainty"]
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

    data = run_parallel(sims) do sim, hist

        return [:steps=>n_steps(hist),
                :reward=>discounted_reward(hist),
                :iterations=>mean(info[:tree_queries] for info in eachstep(hist,"ai"))
               ]
    end
    # data = run(sims)

    rs = data[:reward]
    println(@sprintf("reward: %6.3f ± %6.3f", mean(rs), std(rs)/sqrt(length(rs))))
    is = data[:iterations]
    println(@sprintf("iterations: mean: %6.3f", mean(is)))

    if isempty(alldata)
        alldata = data
    else
        alldata = vcat(alldata, data)
    end
end

p = plot()
for data in groupby(alldata, :solver)
    histogram!(p, data[:steps], label=first(data[:solver]))
end

gui(p)

#=
datestring = Dates.format(now(), "E_d_u_HH_MM")
copyname = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "icaps_2018", "data", "subhunt_table_$(datestring).jl")
write(copyname, file_contents)
filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "icaps_2018", "data", "subhunt_$(datestring).csv")
println("saving to $filename...")
writetable(filename, alldata)
println("done.")
=#
