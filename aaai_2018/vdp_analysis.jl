using DataFrames
using PGFPlots

legend = Dict("pomcpow"=>"POMCPOW",
              "pomcp"=>"POMCP",
              "pft_5"=>"PF Tree (m=5)",
              "pft_10"=>"PF Tree (m=10)",
              "pft_50"=>"PF Tree (m=50)",
              "pft_100"=>"PF Tree (m=100)",
              "pft_1000"=>"PF Tree (m=1000)"
             )

# computation time figure

filename = "vdp_trends_Sunday_10_Sep_04_30.csv"

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

a = Axis(lines, xmode="log", xlabel="Step Computation Limit (s)", ylabel="Mean Discounted Reward", legendPos="south east")
pdf = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "aaai_2018", "vdp_trends.pdf")

save(pdf, a)
run(`xdg-open $pdf`)

# discretization figure

filename = "vdp_discretization_Saturday_9_Sep_14_59.csv"

alldata = readtable(Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "data", filename))

summary = by(alldata, [:solver, :n_angles]) do df
    r = df[:reward]
    DataFrame(reward_mean=mean(r),
              reward_stderr=std(r)/sqrt(length(r)),
             )
end

@show summary

lines = PGFPlots.Plot[]
for df in groupby(summary, [:solver])
    line = PGFPlots.Linear(df[:n_angles],
                           df[:reward_mean],
                           errorBars=ErrorBars(y=df[:reward_stderr]),
                           legendentry=legend[first(df[:solver])]
                          )
    push!(lines, line)
end

a = Axis(lines, xmode="log", ymin=-20, xlabel="Number of Discrete Angles", ylabel="Mean Discounted Reward", legendPos="south east")
pdf = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "aaai_2018", "vdp_discretization.pdf")

save(pdf, a)
run(`xdg-open $pdf`)
