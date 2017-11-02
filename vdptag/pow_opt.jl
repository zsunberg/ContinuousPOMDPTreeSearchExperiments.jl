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
using Distributions
using Query

function gen_sims(x::Vector{Float64}, n, k)
    c = max(0.0, x[1])
    k_act = max(1.0, x[2])
    inv_alpha_act = max(0.1, x[3])
    k_obs = max(1.0, x[4])
    inv_alpha_obs = max(0.1, x[5])

    rng = Base.GLOBAL_RNG
    ro = ToNextMLSolver(rng)
    solver = POMCPOWSolver(tree_queries=10_000_000,
                           criterion=MaxUCB(c),
                           final_criterion=MaxTries(),
                           max_depth=10,
                           max_time=0.1,
                           k_action=k_act,
                           alpha_action=1/inv_alpha_act,
                           k_observation=k_obs,
                           alpha_observation=1/inv_alpha_obs,
                           estimate_value=FORollout(ro),
                           check_repeat_act=false,
                           check_repeat_obs=false,
                           next_action=RootToNextMLFirst(rng),
                           default_action=ReportWhenUsed(1),
                           rng=rng
                          )

    pomdp = VDPTagPOMDP()
    planner = solve(solver, pomdp)

    sims = []
    for i in 1:n
        filter = ObsAdaptiveParticleFilter(deepcopy(pomdp),
                                           LowVarianceResampler(10_000),
                                           0.05,
                                           # MersenneTwister(i+90_000))            
                                           MersenneTwister(rand(UInt32)))

        sim = Sim(deepcopy(pomdp),
                  deepcopy(planner),
                  filter,
                  rng=MersenneTwister(rand(UInt32)),
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

start_mean = [40.0, 8.0, 20.0, 4.0, 20.0]
start_cov = diagm([40.0^2, 10.0^2, 20.0^2, 10.0^2, 20.0^2])
d = MvNormal(start_mean, start_cov)
rng = MersenneTwister(15)
K = 100
m = 30

for i in 1:100
    sims = []
    params = Vector{Vector{Float64}}(K)
    for k in 1:K
        p = rand(d)
        params[k] = p
        k_sims = gen_sims(p, 2, k)
        append!(sims, k_sims)
    end
    results = run_parallel(sims)
    combined = by(results, :k) do df
        DataFrame(mean_reward=mean(df[:reward]))
    end
    @show mean(combined[:mean_reward])
    order = sortperm(combined[:mean_reward])
    elite = params[combined[:k][order[K-m:end]]]
    elite_matrix = Matrix{Float64}(5, m)
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
    @show mean(d)
    @show eigvals(cov(d))
    @show eigvecs(cov(d))
end
