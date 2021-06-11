using Plots: minmax, print
using CSV, DataFrames
using Plots; gr()
using Pipe

df = CSV.File("data/irish_whiskey.csv") |> DataFrame
df_world_pop = CSV.File("data/worldbank_population_total.csv") |> DataFrame


df_world_pop."Indicator Name" |> unique
select!(df_world_pop, vcat(["Country Name"], string.(minimum(df.year):maximum(df.year))))
rename!(df_world_pop, "Country Name" => "country")
transform!(df_world_pop, :country => (c -> c == "Russian Federation" ? "Russia" : c) => :country)
transform!(df_world_pop, :country => (country_v -> [cmp(c, "Russian Federation") == 0 ? "Russia" : c for c in country_v]))

countries_wb = [cmp(c, "Russian Federation") == 0 ? "Russia" : c for c in df_world_pop.country] |> unique
filter(c -> match(r"Russ", c) !== nothing, countries_wb)

countries = filter(r -> match(r"^DF\s", r.country) === nothing, df).country |> unique

countries = unique(df.country)
countries_wb = unique(df_world_pop.country)

for c in countries
    if !(c in countries_wb)
        println(c)
    end
end


filter!(r -> r.country in df.country, df_world_pop)


dropmissing(df).country |> unique |> length
df.country |> unique |> length

df.category |> unique

df.year |> unique

@pipe df.cases |> skipmissing |> collect |> filter(x -> x < 1e5, _) |> histogram

dropmissing!(df)

match(r"^DF\s", "DF xd")

filter(c -> match(r"Russ", c) !== nothing, countries_wb)

"xd" == "xd"