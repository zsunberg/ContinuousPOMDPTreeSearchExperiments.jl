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
using VDPTag2
using QMDP
using MCTS

@everywhere using VDPTag2
@everywhere using POMDPs

pomdp = VDPTagPOMDP(mdp=VDPTagMDP(barriers=CardinalBarriers(0.2, 2.8)))

@show max_time = 1.0
@show max_depth = 10
@show RO = RandomSolver

function gen_sims(x::Vector{Float64}, n, k, seed)
    m = round(Int, x[1])
    @assert m >= 1
    c = x[2]
    @assert c >= 0.0
    k_act = x[3]
    @assert k_act >= 1.0
    inv_alpha_act = x[4]
    @assert inv_alpha_act >= 0.1
    k_obs = x[5]
    @assert k_obs >= 1.0
    inv_alpha_obs = x[6]
    @assert inv_alpha_obs >= 0.1

    sims = []

    for i in 1:n
        rng = MersenneTwister(0)

        node_updater = ObsAdaptiveParticleFilter(deepcopy(pomdp),
                                           LowVarianceResampler(m),
                                           0.05, rng)            
        # ro = ToNextML(mdp(pomdp), rng)
        ro = RandomSolver(rng)::RO
        ev = SampleRollout(solve(ro, pomdp), rng)
        solver = DPWSolver(n_iterations=typemax(Int),
                           exploration_constant=c,
                           depth=max_depth,
                           max_time=max_time,
                           k_action = k_act, 
                           alpha_action = 1/inv_alpha_act,
                           k_state = k_obs,
                           alpha_state = 1/inv_alpha_obs,
                           check_repeat_state=false,
                           check_repeat_action=false,
                           estimate_value=ev,
                           next_action=RootToNextMLFirst(rng),
                           rng=rng
                          )
        belief_mdp = GenerativeBeliefMDP(deepcopy(pomdp), node_updater)
        planner = solve(solver, belief_mdp)

        policy = PolicyWrapper(planner) do planner, s
            local a::TagAction
            try
                a = action(planner, s)
            catch ex
                @show typeof(ex)
                warn("error in policy evaluation; using default")
                a = TagAction(false, 0.0)
            end
            return a::TagAction
        end

        filter = ObsAdaptiveParticleFilter(deepcopy(pomdp),
                                           LowVarianceResampler(10_000),
                                           0.05,
                                           MersenneTwister(i+10000*k))            

        srand(policy, i+40000*k)
        sim = Sim(deepcopy(pomdp),
                  policy,
                  filter,
                  rng=MersenneTwister(i+50_000*k),
                  max_steps=50,
                  metadata=Dict(:i=>i, :k=>k)
                 )
        push!(sims, sim)
    end
    
    return sims
end

# start_mean = [100.0, 2.0, 10.0]
# start_cov = diagm([100.0^2, 10.0^2, 20.0^2])
start_mean =        [14.0,   75.0,   20.0,   20.0,   8.0,   60.0]
start_cov = diagm([8.0^2, 60.0^2,  10.0^2, 10.0^2, 5.0^2, 30.0^2])
d = MvNormal(start_mean, start_cov)
K = 160 # 60 # number of parameter samples
n = 20  # 100 # number of evaluation simulations
m = 40  # 15 # number of elite samples
max_iters = 100

for i in 1:max_iters
    sims = []
    params = Vector{Vector{Float64}}(K)
    print("creating $K simulation sets")
    for k in 1:K
        p = rand(d)
        p[1] = max(1.0, p[1])
        p[2] = max(1.0, p[2])
        p[3] = max(1.0, p[3])
        p[4] = max(1.0, p[4])
        p[5] = max(1.0, p[5])
        p[6] = max(1.0, p[6])
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
    elite = params[combined[:k][order[K-m+1:end]]]
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
