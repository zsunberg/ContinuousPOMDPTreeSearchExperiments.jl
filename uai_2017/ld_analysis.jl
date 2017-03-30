using JLD
using Mustache

# counts = load("data/compare_Friday_24_Mar_00_52.jld", "counts")
# steps = load("data/compare_Friday_24_Mar_00_52.jld", "steps")

# without the 
# counts = load("data/compare_Sunday_26_Mar_17_56.jld", "counts")
# steps = load("data/compare_Sunday_26_Mar_17_56.jld", "steps")

println("Average sim counts per step:")
cts_per_step = Dict{String,Float64}()
for k in keys(counts)
    cts_per_step[k] = sum(counts[k])/sum(steps[k])
    println("\t$k: $(cts_per_step[k])")
end

# @show 50_000*mean(cts_per_step["pomcpow"])/mean(cts_per_step["modified_pomcp"])


# mt"""
# \begin{tabular}{l}
# \toprule
# Algorithm & \$n\$ & Avg. 
# \midrule
# 
# \bottomrule
# \end{tabular}
# """
