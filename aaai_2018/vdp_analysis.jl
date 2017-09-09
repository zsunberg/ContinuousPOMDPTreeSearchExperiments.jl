using DataFrames
using PGFPlots

filename = "vdp_trends_Friday_8_Sep_08_50.csv"

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

legend = Dict("pomcpow"=>"POMCPOW",
              "pft_10"=>"PF Tree (m=10)",
              "pft_100"=>"PF Tree (m=100)",
              "pft_1000"=>"PF Tree (m=1000)"
             )

lines = PGFPlots.Plot[]
for df in groupby(summary, [:solver])
    line = PGFPlots.Linear(df[:time],
                           df[:reward_mean],
                           errorBars=ErrorBars(y=df[:reward_stderr]),
                           legendentry=legend[first(df[:solver])]
                          )
    push!(lines, line)
end

a = Axis(lines, xmode="log", ymin=-20.0, ymax=40.0)
pdf = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "aaai_2018", "vdp_trends.pdf")

save(pdf, a)
run(`xdg-open $pdf`)
