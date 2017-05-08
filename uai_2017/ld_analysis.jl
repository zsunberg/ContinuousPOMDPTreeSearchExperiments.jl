using JLD
using Mustache

# counts = load("data/compare_Friday_24_Mar_00_52.jld", "counts")
# steps = load("data/compare_Friday_24_Mar_00_52.jld", "steps")

# without the 
# counts = load("data/compare_Sunday_26_Mar_17_56.jld", "counts")
# steps = load("data/compare_Sunday_26_Mar_17_56.jld", "steps")

# solver_keys = load("data/compare_Wednesday_29_Mar_10_05.jld", "solver_keys")
counts = load("data/compare_Wednesday_29_Mar_10_05.jld", "counts")
rewards = load("data/compare_Wednesday_29_Mar_10_05.jld", "rewards")
steps = load("data/compare_Wednesday_29_Mar_10_05.jld", "steps")

for k in keys(counts)
    # cts_per_step[k] = sum(counts[k])/sum(steps[k])
    rew = rewards[k]
    sem = std(rew)/sqrt(length(rew))
    cps = sum(counts[k])/sum(steps[k])
    println("$k & $(mean(rew)) \\pm $sem & $cps")
end

counts = load("data/compare_Wednesday_29_Mar_19_04.jld", "counts")
rewards = load("data/compare_Wednesday_29_Mar_19_04.jld", "rewards")
steps = load("data/compare_Wednesday_29_Mar_19_04.jld", "steps")

for k in keys(counts)
    # cts_per_step[k] = sum(counts[k])/sum(steps[k])
    rew = rewards[k]
    sem = std(rew)/sqrt(length(rew))
    cps = sum(counts[k])/sum(steps[k])
    println("$k & $(mean(rew)) \\pm $sem & $cps")
end

times = load("data/compare_Friday_31_Mar_20_23.jld", "times")
steps = load("data/compare_Friday_31_Mar_20_23.jld", "steps")

for k in keys(times)
    tps = sum(times[k])/sum(steps[k])
    println("$k: $tps")
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
