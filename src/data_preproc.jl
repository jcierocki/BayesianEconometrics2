using CSV, DataFrames, ShiftedArrays
using Plots; gr()
using Pipe
using Statistics, Distributions, Random
using Base: parse

Random.seed!(1234)

#### data processing

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

df_pop = DataFrames.stack(df_world_pop, Not(:country), variable_name = :year, value_name = :population)
transform!(df_pop, :year => (y -> parse.(Int, y)) => :year)

pop_dict = Dict(zip(df_pop.country, df_pop.year) |> collect .=> df_pop.population)
transform!(df, [:country, :year, :cases] => ByRow((c, y, k) -> (k / pop_dict[(c, y)]) * 1_000_000) => :popularity)
select!(df, Not(:cases))

### Adding ommited NA's

for c in unique(df.country), q in unique(df.quality), y in unique(df.year)
    if filter(r -> r.country == c && r.quality == q && r.year == y, df) |> nrow == 0
        push!(df, Dict(:country => c, :quality => q, :year => y, :popularity => missing))
    end
end

sort!(df, [:country, :quality, :year])

###

### number of non-NA's

@pipe df |>
    groupby(_, [:country, :quality]) |>
    combine(_, :popularity => (x -> skipmissing(x) |> collect |> length) => :count) |>
    sort(_, :count) |>
    first(_, 40) |>
    print

@pipe df |>
    dropmissing |>
    groupby(_, [:quality]) |>
    combine(_, nrow => :count) |>
    sort(_, :count)

@pipe df |>
    dropmissing |>
    groupby(_, [:year]) |>
    combine(_, nrow => :count) |>
    sort(_, :count)

@pipe df |>
    dropmissing |>
    groupby(_, [:country]) |>
    combine(_, nrow => :count) |>
    sort(_, :count) |>
    first(_, 10)

df.country |> unique |> length

## We're going to skip Mexico beacuse of very few data, making it unable to calculate variance
## Futher (after lagging) we're also going to skip Super Premium for , Finland, Hungary and Lativia from the similiar reason

filter!(r -> r.country != "Mexico", df)

###

df.popularity |> skipmissing |> collect .|> log |> histogram

df_transformed = @pipe df |>
    groupby(_, [:country, :quality]) |>
    transform(_, :popularity => lag => :popularity_lag) |>
    vcat

filter!(r -> !(r.country in ["Switzerland", "Finland", "Hungary", "Lativia"] && r.quality == "Super Premium"), df_transformed)
filter!(r -> !ismissing(r.popularity), df_transformed)

## we will ommit Brazil while converting to binary

for q in unique(df_transformed.quality)
    transform!(df_transformed, :quality => (x -> Float64.(x .== q)) => q)
end

for c in unique(df_transformed.country)
    transform!(df_transformed, :country => (x -> Float64.(x .== c)) => c)
end

select!(df_transformed, Not([:Standard, :Brazil]))

rename!(df_transformed, "popularity" => "y", "popularity_lag" => "y_lag")
transform!(df_transformed, [:y, :y_lag] .=> (x -> log.(x .+ 1)) .=> [:y, :y_lag])

CSV.write("data/transformed_data.csv", df_transformed)