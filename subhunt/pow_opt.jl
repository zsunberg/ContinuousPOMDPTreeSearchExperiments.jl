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
using SubHunt
using QMDP

@everywhere begin
    using POMDPs
    using ParticleFilters
    using POMDPToolbox
    using ContinuousPOMDPTreeSearchExperiments
end

pomdp = SubHuntPOMDP()
vs = ValueIterationSolver()
if !isdefined(:vp) || vp.mdp != pomdp
    vp = solve(vs, pomdp, verbose=true)
end
qs = QMDPSolver()
qp = QMDP.create_policy(qs, pomdp)
qp.alphas[:] = vp.qmat


function gen_sims(x::Vector{Float64}, n, k, seed)
    c = x[1]
    @assert c >= 0.0
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
                               max_depth=20,
                               max_time=1.0,
                               enable_action_pw=false,
                               k_observation=k_obs,
                               alpha_observation=1/inv_alpha_obs,
                               estimate_value=FOValue(vp),
                               check_repeat_obs=false,
                               default_action=ReportWhenUsed(qp),
                               rng=rng
                              )


        planner = solve(solver, pomdp)
        filter = ObsAdaptiveParticleFilter(deepcopy(pomdp),
                                           LowVarianceResampler(10_000),
                                           0.05,
                                           MersenneTwister(i+10000*k))            

        srand(planner, i+40000*k)
        sim = Sim(deepcopy(pomdp),
                  planner,
                  filter,
                  rng=MersenneTwister(i+50_000*k),
                  max_steps=100,
                  metadata=Dict(:i=>i, :k=>k)
                 )
        push!(sims, sim)
    end
    
    return sims
end

# start_mean = [100.0, 2.0, 10.0]
# start_cov = diagm([100.0^2, 10.0^2, 20.0^2])
start_mean = [17.0, 6.0, 50.0]
start_cov = diagm([20.0^2, 10.0^2, 50.0^2])
d = MvNormal(start_mean, start_cov)
K = 120 # 60 # number of parameter samples
n = 100 # 100 # number of evaluation simulations
m = 15  # 15 # number of elite samples
max_iters = 100

for i in 1:max_iters
    sims = []
    params = Vector{Vector{Float64}}(K)
    print("creating $K simulation sets")
    for k in 1:K
        p = rand(d)
        p[1] = max(0.0, p[1])
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
    @show det(cov(d))
    @show ev = eigvals(cov(d))
    for j in 1:length(ev)
        @show eigvecs(cov(d))[:,j]
    end
end
