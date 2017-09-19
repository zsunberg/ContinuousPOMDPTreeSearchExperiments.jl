using DataFrames
using PGFPlots
using Query

legend = Dict("pomcpow"=>"POMCPOW",
              "pomcp"=>"POMCP",
              "pft_5"=>"PF Tree (m=5)",
              "pft_10"=>"PF Tree (m=10)",
              "pft_50"=>"PF Tree (m=50)",
              "pft_100"=>"PF Tree (m=100)",
              "pft_1000"=>"PF Tree (m=1000)"
             )

# computation time figure

filename = "vdp_trends_Monday_11_Sep_19_20.csv"

alldata = readtable(Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "data", filename))

summary = by(alldata, [:solver, :time]) do df
    r = df[:reward]
    mean_cpu = df[:mean_cpu_us]
    valid_mean_cpu = mean_cpu[find(.!isna.(mean_cpu))]
    mean_nodes = df[:mean_nodes]
    valid_mean_nodes = mean_nodes[find(.!isna.(mean_nodes))]
    DataFrame(reward_mean=mean(r),
              reward_stderr=std(r)/sqrt(length(r)),
              cpu=mean(valid_mean_cpu)/1_000_000,
              nodes=mean(valid_mean_nodes)
             )
end

lines = PGFPlots.Plot[]
for df in groupby(summary, [:solver])
    line = PGFPlots.Linear(df[:time],
                           df[:reward_mean],
                           errorBars=ErrorBars(y=df[:reward_stderr]),
                           legendentry=legend[first(df[:solver])]
                          )
    push!(lines, line)
end

cheight = get(first(@from i in summary begin
    @where i.time == 0.1 && i.solver == "pomcpow"
    @select i.reward_mean
end))
@show cheight
# warn("cheight")
# cheight = 38.26

a = Axis(lines, xmode="log", xlabel="Step Computation Limit (s)", ylabel="Mean Discounted Reward", legendPos="south east")
pdf = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "aaai_2018", "vdp_trends.pdf")

tex = tempname()*".tex"
save(tex, a, include_preamble=false)
run(`cat $tex`)
println("\n\n\n")

save(pdf, a)
run(`xdg-open $pdf`)

# discretization figure

filename = "vdp_discretization_Monday_11_Sep_13_28.csv"

legend = Dict("pomcpow"=>"POMCPOW (Discretized)",
              "pomcp"=>"POMCP (Discretized)",
             )


alldata = readtable(Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "data", filename))

summary = by(alldata, [:solver, :n_angles]) do df
    r = df[:reward]
    DataFrame(reward_mean=mean(r),
              reward_stderr=std(r)/sqrt(length(r)),
             )
end

lines = PGFPlots.Plot[]
for df in groupby(summary, [:solver])
    line = PGFPlots.Linear(df[:n_angles],
                           df[:reward_mean],
                           errorBars=ErrorBars(y=df[:reward_stderr]),
                           legendentry=legend[first(df[:solver])]
                          )
    push!(lines, line)
end
ends = vcat(minimum(alldata[:n_angles]), maximum(alldata[:n_angles]))
cline = PGFPlots.Linear(ends, [cheight, cheight], style="ultra thick, dashed", mark="none", legendentry="POMCPOW (Continuous)")
push!(lines, cline)

a = Axis(lines, xmode="log", ymin=-5, xlabel="Number of Discrete Action and Observation Angles", ylabel="Mean Discounted Reward", legendPos="south east")
pdf = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "aaai_2018", "vdp_discretization.pdf")

tex = tempname()*".tex"
save(tex, a, include_preamble=false)
run(`cat $tex`)

save(pdf, a)
run(`xdg-open $pdf`)
