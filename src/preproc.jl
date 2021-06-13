using CSV, DataFrames, Pipe
using ShiftedArrays
using Statistics, Random
using Base: parse

Random.seed!(1234)

####

df = CSV.File("data/irish_whiskey.csv") |> DataFrame
df_world_pop = CSV.File("data/worldbank_population_total.csv") |> DataFrame

df.category |> unique # only one value => column can be ommited
select!(df, Not(:category))

select!(df_world_pop, vcat(["Country Name"], string.(minimum(df.year):maximum(df.year)))) # filtering by time
rename!(df_world_pop, "Country Name" => "country")

## suming population of Belgium and Luxembourg as those countries presented together in original dataset and unable to split
belgium_and_luxembourg = @pipe df_world_pop |>
    filter(r -> r.country in ["Belgium", "Luxembourg"], _) |>
    combine(_, :country => (x -> "Belgium and Luxembourg") => :country, Symbol.(1990:2016) .=> sum .=> Symbol.(1990:2016)) |>
    DataFrameRow(_, 1)

push!(df_world_pop, belgium_and_luxembourg)

## Restricting to EU14 + US + Canada
countries_eu14_america = [
    "Belgium and Luxembourg",
    "France",
    "Germany",
    "Italy",
    "Netherlands",
    "Denmark",
    "Ireland",
    "United Kingdom",
    "Greece",
    "Portugal",
    "Spain",
    "Austria",
    "Finland",
    "Sweden",
    "United States",
    "Canada",
]

filter!(r -> r.country in countries_eu14_america, df)

## Scaling using The World Bank demographic data
df_pop = DataFrames.stack(df_world_pop, Not(:country), variable_name = :year, value_name = :population)
transform!(df_pop, :year => (y -> parse.(Int, y)) => :year)

pop_dict = Dict(tuple.(df_pop.country, df_pop.year) .=> df_pop.population)
transform!(df, [:country, :year, :cases] => ByRow((c, y, k) -> (k / pop_dict[(c, y)]) * 1_000_000) => :popularity)
select!(df, Not(:cases))

## Adding ommited NA's
for c in unique(df.country), q in unique(df.quality), y in unique(df.year)
    if filter(r -> r.country == c && r.quality == q && r.year == y, df) |> nrow == 0
        push!(df, Dict(:country => c, :quality => q, :year => y, :popularity => missing))
    end
end

sort!(df, [:country, :quality, :year])

## Removing country-quality pairs without sufficient amout of non-missing observations
@pipe df |>
    groupby(_, [:country, :quality]) |>
    combine(_, :popularity => (x -> skipmissing(x) |> collect |> length) => :count) |>
    sort(_, :count) |>
    first(_, 20)

filter!(r -> r.country != "Greece", df)
filter!(r -> !(r.quality == "Super Premium" && r.country in ["Sweden", "Spain", "Belgium and Luxembourg", "Finland"]), df)

## Adding lagged target variable
df_transformed = @pipe df |>
    groupby(_, [:country, :quality]) |>
    transform(_, :popularity => lag => :popularity_lag) |>
    vcat

filter!(r -> !ismissing(r.popularity), df_transformed)

## Choosing reference country to be ommited while creating binaries
@pipe df_transformed |>
    groupby(_, :country) |>
    combine(_, :popularity => mean => :mean) |>
    sort(_, :mean) |>
    first(_, 5)

select!(df_transformed, [:country, :quality, :year, :popularity, :popularity_lag])

## Saving file
CSV.write("data/transformed_data.csv", df_transformed)