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
using DiscreteValueIteration

@everywhere begin
    using POMDPs
    using ParticleFilters
    using POMDPToolbox
    using ContinuousPOMDPTreeSearchExperiments
end

pomdp = SimpleLightDark()

function gen_sims(x::Vector{Float64}, n, k, seed)
    c = max(0.0, x[1])
    k_obs = x[2]
    @assert k_obs >= 1.0
    inv_alpha_obs = x[3]
    @assert inv_alpha_obs >= 0.1

    sims = []

    for i in 1:n
        rng = MersenneTwister(0)
        ro = ValueIterationSolver()
        solver = POMCPOWSolver(tree_queries=10_000_000,
                               criterion=MaxUCB(c),
                               final_criterion=MaxTries(),
                               max_depth=90,
                               max_time=1.0,
                               k_observation=k_obs,
                               alpha_observation=1/inv_alpha_obs,
                               estimate_value=FOValue(ro),
                               enable_action_pw=false,
                               check_repeat_obs=true,
                               # default_action=ReportWhenUsed(TagAction(false, 0.0)),
                               # default_action=TagAction(false, 0.0),
                               rng=rng
                              )

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

start_mean = [100.0, 4.0, 10.0]
start_cov = diagm([40.0^2, 10.0^2, 10.0^2])
d = MvNormal(start_mean, start_cov)
K = 60  # 60
n = 100 # 100
m = 15  # 15
max_iters = 100

for i in 1:max_iters
    sims = []
    params = Vector{Vector{Float64}}(K)
    print("creating $K simulation sets")
    for k in 1:K
        p = rand(d)
        p[2] = max(1.0, p[2])
        p[3] = max(0.1, p[3])
        params[k] = p
        k_sims = gen_sims(p, n, k, i)
        print(".")
        append!(sims, k_sims)
    end
    println()
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
    @show ev = eigvals(cov(d))
    for j in 1:length(ev)
        @show eigvecs(cov(d))[:,j]
    end
end
