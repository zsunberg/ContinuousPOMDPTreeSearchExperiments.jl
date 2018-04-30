using DataFrames
using CSV

data = CSV.read("data/multilane_Saturday_28_Apr_16_17.csv")

means = by(data, :solver) do df
    n = size(df, 1)
    return DataFrame(reward=mean(df[:reward]),
                     reward_sem=std(df[:reward])/sqrt(n)
                    )
end

println(means)
