using POMDPs
using ContinuousPOMDPTreeSearchExperiments
using POMDPToolbox
using BasicPOMCP
using POMCPOW
using ParticleFilters
using CPUTime
using DataFrames
using ARDESPOT
using Distributions
using LaserTag
using DiscreteValueIteration

function gen_sims(x::Vector{Float64}, n, k)
    c = max(0.0, x[1])
    k_obs = max(1.0, x[2])
    inv_alpha_obs = max(0.1, x[3])

    rng = MersenneTwister(0)
    ro = ValueIterationSolver()
    solver = POMCPOWSolver(tree_queries=10_000_000,
                           criterion=MaxUCB(c),
                           final_criterion=MaxTries(),
                           max_depth=90,
                           max_time=1.0,
                           k_observation=k_obs,
                           alpha_observation=1/inv_alpha_obs,
                           estimate_value=FORollout(ro),
                           enable_action_pw=false,
                           check_repeat_obs=true,
                           # default_action=ReportWhenUsed(TagAction(false, 0.0)),
                           # default_action=TagAction(false, 0.0),
                           rng=rng
                          )


    sims = []
    for i in 1:n
        pomdp = gen_lasertag(rng=MersenneTwister(i+70_000*k))
        planner = solve(solver, pomdp)
        filter = ObsAdaptiveParticleFilter(deepcopy(pomdp),
                                           LowVarianceResampler(10_000),
                                           0.05,
                                           MersenneTwister(i+10000*k))            

        srand(planner, i+40000*k)
        sim = Sim(deepcopy(pomdp),
                  deepcopy(planner),
                  filter,
                  rng=MersenneTwister(i+50_000*k),
                  max_steps=100,
                  metadata=Dict(:i=>i, :k=>k)
                 )

        push!(sims, sim)
    end
    
    return sims
end

#=
gen_syms(x::Vector{Float64}, n, k) = Any[k=>copy(x) for i in 1:n]

function run_parallel(sims::Vector{Any})
    ks = Int[]
    rewards = Float64[]
    for (k, v) in sims
        push!(ks, k)
        push!(rewards, -sum(x->x^2, v - [40.0, 30.0, 20.0, 3.1, 28.0]) + 10.0*rand())
    end
    return DataFrame(k=ks, reward=rewards)
end
=#

start_mean = [20.0, 4.0, 10.0]
start_cov = diagm([40.0^2, 10.0^2, 10.0^2])
d = MvNormal(start_mean, start_cov)
rng = MersenneTwister(15)
K = 80  # 200
n = 40  # 40
m = 20  # 50
max_iters = 2

for i in 1:max_iters
    sims = []
    params = Vector{Vector{Float64}}(K)
    for k in 1:K
        p = rand(d)
        params[k] = p
        k_sims = gen_sims(p, n, k)
        println("appending $(length(k_sims)) simulations")
        append!(sims, k_sims)
    end
    results = run_parallel(sims)
    # results = run(sims)
    combined = by(results, :k) do df
        DataFrame(mean_reward=mean(df[:reward]))
    end
    @show mean(combined[:mean_reward])
    order = sortperm(combined[:mean_reward])
    elite = params[combined[:k][order[K-m:end]]]
    elite_matrix = Matrix{Float64}(length(start_mean), m)
    for k in 1:m
        elite_matrix[:,k] = elite[k]
    end
    try
        d = fit(typeof(d), elite_matrix)
    catch ex
        if ex isa Base.LinAlg.PosDefException
            println("pos def exception")
            d = fit(typeof(d), elite_matrix += 0.01*randn(size(elite_matrix)))
        else
            rethrow(ex)
        end
    end
    println("iteration $i")
    @show mean(d)
    @show eigvals(cov(d))
    @show eigvecs(cov(d))
end
