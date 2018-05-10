using DataFrames
using DataFramesMeta
using CSV
using Iterators

solver_order = ["pomcpow" => "POMCPOW",
                "pft" => "PFT-DPW",
                "qmdp" => "QMDP",
                "pomcpdpw" => "POMCP-DPW", 
                "despot" => "DESPOT",
                "d_pomcp" => "POMCP\\textsuperscript{D}",
                "d_despot" => "DESPOT\\textsuperscript{D}"]

cardinality = Dict("lasertag" => "(D, D, D)",
                   "lightdark" => "(D, D, C)",
                   "subhunt" => "(D, D, C)",
                   "vdpbarrier" => "(C, C, C)",
                   "vdptag" => "(C, C, C)",
                   "multilane" => "(C, D, C)"
                  )


problem_order = [["lasertag", "lightdark", "subhunt"], ["vdpbarrier", "multilane"]]
# problem_order = ["multilane"]

filenames = Dict("simpleld"=>"/home/zach/.julia/v0.6/ContinuousPOMDPTreeSearchExperiments/icaps_2018/data/simpleld_Monday_5_Mar_21_39.csv",
                 "subhunt"=>"/home/zach/.julia/v0.6/ContinuousPOMDPTreeSearchExperiments/icaps_2018/data/subhunt_Monday_5_Mar_22_08.csv",
                 "lasertag"=>"/home/zach/.julia/v0.6/ContinuousPOMDPTreeSearchExperiments/icaps_2018/data/lasertag_Monday_5_Mar_20_17.csv",
                 "vdpbarrier"=>"/home/zach/.julia/v0.6/ContinuousPOMDPTreeSearchExperiments/icaps_2018/data/vdpbarrier_Monday_5_Mar_23_01.csv",
                 "multilane"=>"/home/zach/.julia/v0.6/ContinuousPOMDPTreeSearchExperiments/data/multilane_Monday_30_Apr_23_41.csv"
                )
filenames["lightdark"] = filenames["simpleld"]

# filenames = Dict("lasertag" => "$(Pkg.dir("ContinuousPOMDPTreeSearchExperiments"))/icaps_2018/data/lasertag_Monday_26_Feb_18_46.csv",
#                  "lightdark" => "$(Pkg.dir("ContinuousPOMDPTreeSearchExperiments"))/icaps_2018/data/simpleld_Monday_26_Feb_20_13.csv",
#                  "subhunt" => "$(Pkg.dir("ContinuousPOMDPTreeSearchExperiments"))/icaps_2018/data/subhunt_Monday_26_Feb_20_44.csv",
#                  "vdpbarrier" => "$(Pkg.dir("ContinuousPOMDPTreeSearchExperiments"))/icaps_2018/data/bdpbarrier_Monday_26_Feb_21_42.csv")

data = Dict(
    "lightdark" => Dict(
        "limits" => (-20.0, 80.0),
        "name" => "Light Dark"
    ), 

    "subhunt" => Dict(
        "limits" => (0.0, 80.0),
        "name" => "Sub Hunt"
    ),

    "lasertag" => Dict(
        "limits" => (-20.0, -8.0),
        "name" => "Laser Tag"
    ),

    "vdptag" => Dict(
        "limits" => (-20.0, 40.0),
        "name" => "VDP Tag"
    ),

    "vdpbarrier" => Dict(
        "limits" => (0.0, 31.0),
        "name" => "VDP Tag"
    ),

    "multilane" => Dict(
        "limits" => (10.0, 40.0),
        "name" => "Multilane",
    )

)

for p in chain(problem_order...)
    df = CSV.read(filenames[p])
    d = data[p]
    for s in unique(df[:solver])
        rs = @where(df, :solver.==s)[:reward]
        m = mean(rs)
        sem = std(rs)/sqrt(length(rs))
        if s == "ar_despot"
            s = "despot"
        elseif s == "despot_01"
            s = "despot"
        elseif p == "lasertag" && s == "pomcp"
            s = "d_pomcp"
        elseif p == "multilane" && s == "pftdpw"
            s = "pft"
        elseif p == "multilane" && s == "qmdp"
            s = "pomcpdpw"
        end
        if p == "multilane"
            m *= 100.0
            sem *= 100.0
        end
        d[s] = (m, sem)
    end
end

hbufs = []
tbufs = []

for po_chunk in problem_order
    hbuf = IOBuffer()

    for k in po_chunk
        d = data[k]
        print(hbuf, "& & $(d["name"]) \\makebox[0pt][l]{$(cardinality[k])} & ")
    end
    print(hbuf, "\\\\")

    tbuf = IOBuffer()
    for (k, n) in solver_order
        print(tbuf, n*" ")
        for p in po_chunk
            d = data[p]
            if haskey(d, k)
                m, sem = d[k]
                lo, hi = d["limits"]
                frac = (m-lo)/(hi-lo)
                @printf(tbuf, "& & \\result{%.1f}{%.1f}{%d}{%.2f} ",
                        m, sem, round(Int, 100*frac), frac
                       )
            else
                print(tbuf,  "& & \\noresult{} ")
            end
        end
        print(tbuf, "\\\\\n")
    end

    push!(hbufs, hbuf)
    push!(tbufs, tbuf)
end

tablebuf = IOBuffer()

columns = "l"*"Xrl"^maximum(length(chunk) for chunk in problem_order)

println(tablebuf, "\\begin{tabularx}{\\linewidth}{$columns}")
println(tablebuf, "\\toprule")

for i in 1:length(hbufs)
    println(tablebuf, String(hbufs[i]))
    println(tablebuf, "\\midrule")
    println(tablebuf, String(tbufs[i]))

    if i < length(hbufs)
        println(tablebuf, "\\midrule")
    else
        println(tablebuf, "\\bottomrule")
        println(tablebuf, "\\end{tabularx}")
    end
end

open("/home/zach/Devel/thesis/results_table.tex", "w") do f
    println(f, String(tablebuf))
end
