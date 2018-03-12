using PGFPlots
using CSV
using Query
using DataFrames

# ld
filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "icaps_2018", "data", "ld_discretization_Tuesday_27_Feb_15_23.csv")
alldata = CSV.read(filename)

aggregated = by(alldata, [:solver, :d]) do df
    r = df[:reward]
    return DataFrame(mean=mean(r), sem=std(r)/sqrt(length(r)))
end


names = Dict("pomcpow"=>"POMCPOW\\textsuperscript{D}",
             "d_pomcp"=>"POMCP\\textsuperscript{D}",
             "d_despot"=>"DESPOT\\textsuperscript{D}")
plots = Plots.Plot[]
for k in ["d_pomcp", "d_despot", "pomcpow"]
    series = @from pt in aggregated begin
        @where pt.solver == k
        @select {pt.d, pt.mean, pt.sem}
        @collect DataFrame
    end
    push!(plots, Plots.Linear(series, x=:d, y=:mean, errorBars=ErrorBars(y=series[:sem]), legendentry=names[k]))
end
ds = sort(unique(aggregated[:d]))
push!(plots, Plots.Linear(ds,
                          fill!(similar(ds), 57.7),
                          errorBars=ErrorBars(y=fill!(similar(ds), 0.5)),
                          legendentry="POMCPOW\\phantom{\\textsuperscript{D}}"))
p = plot(Axis(plots,
              xlabel="Discretization bin size",
              ylabel="Mean accumulated reward",
              xmode="log",
              style="legend style={at={(1.06,0.3)},anchor=south east,font=\\small}"))
save("/tmp/ld_discretization.pdf", p)

# Subhunt

filename = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "icaps_2018", "data", "subhunt_discretization_Tuesday_27_Feb_11_38.csv")
alldata = CSV.read(filename)

aggregated = by(alldata, [:solver, :binsize]) do df
    r = df[:reward]
    return DataFrame(mean=mean(r), sem=std(r)/sqrt(length(r)))
end

# println(aggregated)

names = Dict("pomcpow"=>"POMCPOW\\textsuperscript{D}",
             "d_pomcp"=>"POMCP\\textsuperscript{D}",
             "d_despot"=>"DESPOT\\textsuperscript{D}")
plots = Plots.Plot[]
for k in ["d_pomcp", "d_despot", "pomcpow"]
    series = @from pt in aggregated begin
        @where pt.solver == k
        @select {pt.binsize, pt.mean, pt.sem}
        @collect DataFrame
    end
    push!(plots, Plots.Linear(series, x=:binsize, y=:mean, errorBars=ErrorBars(y=series[:sem]), legendentry=names[k]))
end
ds = sort(unique(aggregated[:binsize]))
push!(plots, Plots.Linear(ds,
                          fill!(similar(ds), 69.2),
                          errorBars=ErrorBars(y=fill!(similar(ds), 1.3)),
                          legendentry="POMCPOW\\phantom{\\textsuperscript{D}}"))
p = plot(Axis(plots,
              ymin=0,
              xmode="log",
              xlabel="Discretization bin size",
              ylabel="Mean accumulated reward",
              style="legend style={at={(0.01,0.45)},anchor=south west,font=\\small}"))
save("/tmp/subhunt_discretization.pdf", p)
