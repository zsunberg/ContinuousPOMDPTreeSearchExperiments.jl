using Plots
using ContinuousPOMDPTreeSearchExperiments
using POMDPs
using DiscreteValueIteration
using POMCPOW
using ParticleFilters
using POMDPToolbox

pyplot()

@show max_time = 1.0
@show max_depth = 20

pomdp = SimpleLightDark()

solvers = Dict{String, Union{Solver,Policy}}(
    "pomcpow" => begin
        rng = MersenneTwister(13)
        ro = ValueIterationSolver()
        solver = POMCPOWSolver(tree_queries=10_000_000,
                               criterion=MaxUCB(90.0),
                               # final_criterion=MaxTries(),
                               max_depth=max_depth,
                               max_time=max_time,
                               enable_action_pw=false,
                               k_observation=5.0,
                               alpha_observation=1/15.0,
                               estimate_value=FOValue(ro),
                               check_repeat_obs=false,
                               # default_action=ReportWhenUsed(-1),
                               rng=rng
                              )
    end,

    "pomcpdpw" => begin
        rng = MersenneTwister(15)
        ro = ValueIterationSolver()
        solver = PDPWSolver(tree_queries=10_000_000,
                            c=100.0,
                            max_depth=max_depth,
                            max_time=max_time,
                            enable_action_pw=false,
                            k_observation=4.0,
                            alpha_observation=1/10,
                            estimate_value=FOValue(ro),
                            check_repeat_obs=false,
                            # default_action=ReportWhenUsed(-1),
                            rng=rng
                           )
    end
   )

names = Dict("pomcpow"=>"POMCPOW", "pomcpdpw"=>"POMCP-DPW (≈QMDP)")
if !isdefined(:results) || isempty(results)
    results = Dict{String, Any}()
end

for (k, solver) in solvers
    if !haskey(results, k)
        planner = solve(solver, pomdp)
        rng = MersenneTwister(7)
        sim = HistoryRecorder(show_progress=true, max_steps=100, rng=rng, initial_state=3)
        filter = ObsAdaptiveParticleFilter(deepcopy(pomdp),
                                           LowVarianceResampler(1_000),
                                           0.05, MersenneTwister(90_000)
                                          )            
        hist = simulate(sim, pomdp, planner, filter)
        results[k] = hist
    end
end

smin = -10
smax = 20


tmax = maximum(length(state_hist(h)) for h in values(results))
plots = []
p = nothing
for (k, hist) in results
    vsh = collect(filter(s->!isterminal(pomdp,s), state_hist(hist)))
    bh = belief_hist(hist)[1:end-1]

    pts = Int[]
    pss = Int[]
    pws = Float64[]

    for t in 0:length(bh)-1
        b = bh[t+1]
        for s in smin:smax
            w = 10.0*sqrt(pdf(b, s))
            if 0.0<w<1.0
                w = 1.0
            end
            push!(pts, t)
            push!(pss, s)
            push!(pws, w)
        end
    end

    T = linspace(0.0, tmax)
    S = linspace(-1.0, 21.0)
    inv_grays = cgrad([RGB(1.0, 1.0, 1.0),RGB(0.0,0.0,0.0)])
    p = contour(T, S, (t,s)->abs(s-pomdp.light_loc),
            bg_inside=:black,
            fill=true,
            xlim=(0, tmax),
            ylim=(smin, smax),
            color=inv_grays,
            ylabel="State",
            cbar=false,
            legend=:topright
           )
    plot!(p, [0, tmax], [0, 0], linewidth=2, color="green", label="Goal", line=:dash)
    scatter!(p, pts, pss, color="lightblue", label="Belief Particles", markersize=pws, marker=stroke(0.1, 0.3))
    plot!(p, 0:length(vsh)-1, vsh, linewidth=3, color="orangered", label="Trajectory", title=names[k])
    ffamily = "Times"
    smfont = Plots.font(ffamily, 12)
    mfont = Plots.font(ffamily, 12)
    tfont = Plots.font(ffamily, 16)
    plot!(p, titlefont=tfont, legendfont=mfont, tickfont=smfont, guidefont=mfont)
    push!(plots, p)
end

xlabel!(p, "Time")
plot!(p, legend=false)

plot(plots...; layout=(length(plots),1))
gui()
