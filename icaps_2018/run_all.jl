filenames = Dict{String, String}()

tests = ["lasertag",
         "simpleld",
         "subhunt",
         "vdpbarrier"]

for t in tests
    try
        println("\n$(uppercase(t))\n")
        include(t*"_table.jl")
        filenames[t] = filename
    catch ex
        showerror(STDERR, ex)
    end
end

# for dt in ["subhunt_discretization.jl", "ld_discretization.jl"]
#     try
#         println("\n$(uppercase(dt))\n")
#         include(dt)
#         filenames[dt] = filename
#     catch ex
#         showerror(STDERR, ex)
#     end
# end

println("\nFILENAMES\n")
for (k, v) in filenames
    println("$k => $v")
end

@show filenames
