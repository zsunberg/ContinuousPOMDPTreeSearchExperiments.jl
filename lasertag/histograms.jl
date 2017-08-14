using Plots
using JLD

plotly()

N=100
for f in ARGS
    ext = last(split(f, '.'))
    if ext == "jld"
        rewards = load(f, "rdict")["despot"]
    elseif ext == "txt"
        rewards = readdlm(f)
    else
        warn("Unrecognized file type: $f")
    end
    histogram!(rewards[1:N], title=f)
    println(f)
    println("rewards < -19: $(find(rewards.<-19.0))")
end

gui()
