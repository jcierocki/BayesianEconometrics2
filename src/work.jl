using CategoricalArrays: convert
using CSV: stricterror
using Base: typesof, Float64
using Plots: minmax, print
using CSV, DataFrames, ShiftedArrays, CategoricalArrays
using Plots; gr()
using Pipe
using Statistics

df = CSV.File("data/irish_whiskey.csv") |> DataFrame
df_world_pop = CSV.File("data/worldbank_population_total.csv") |> DataFrame

df_world_pop."Indicator Name" |> unique ## only one value => column can be ommited
select!(df_world_pop, vcat(["Country Name"], string.(minimum(df.year):maximum(df.year))))
rename!(df_world_pop, "Country Name" => "country")
transform!(df_world_pop, :country => ByRow(c -> c == "Russian Federation" ? "Russia" : c) => :country)
transform!(df_world_pop, :country => ByRow(c -> c == "Slovak Republic" ? "Slovakia" : c) => :country)

countries_wb = unique(df_world_pop.country)
filter!(r -> r.country in countries_wb, df)
select!(df, Not(:category))

df_pop = stack(df_world_pop, Not(:country), variable_name = :year, value_name = :population)
transform!(df_pop, :year => (y -> parse.(Int, y)) => :year)

pop_dict = Dict(zip(df_pop.country, df_pop.year) |> collect .=> df_pop.population)
transform!(df, [:country, :year, :cases] => ByRow((c, y, k) -> (k / pop_dict[(c, y)]) * 1_000_000) => :popularity)
select!(df, Not(:cases))

# transform!(df, :country => CategoricalArray => :country)
# transform!(df, :quality => CategoricalArray => :quality)
# describe(df)

df.popularity |> skipmissing |> collect .|> log |> histogram

df_transformed = @pipe df |>
    groupby(_, :country) |>
    transform(_, :popularity => lag => :popularity_lag) |>
    vcat

@pipe df_transformed |>
    groupby(_, :country) |>
    combine(_, :popularity => (x -> mean(skipmissing(x))) => :mean_popularity) |>
    sort(_, :mean_popularity)

## we will ommit Brazil while converting to binary

for q in unique(df_transformed.quality)
    transform!(df_transformed, :quality => (x -> Float64.(x .== q)) => q)
end

for c in unique(df_transformed.country)
    transform!(df_transformed, :country => (x -> Float64.(x .== c)) => c)
end

select!(df_transformed, Not([:Standard, :Brazil]))

y = df_transformed.popularity
A = @pipe df_transformed |>
    select(_, Not([:quality, :country, :year, :popularity])) |>
    Matrix

    