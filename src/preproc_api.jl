using CSV, DataFrames, Pipe, CategoricalArrays
using Statistics, Random

Random.seed!(1234)

### Functions

function prepare_model_df(df_raw::DataFrame, reference_country::Symbol)
    df = transform(df_raw, :quality => (x -> (categorical(x; levels = ["Standard", "Premium", "Super Premium"]) .|> levelcode) .- 2.0) => :quality)

    for c in unique(df.country)
        transform!(df, :country => (x -> Float64.(x .== c)) => c)
    end

    rename!(df, "popularity" => "y", "popularity_lag" => "y_lag")
    transform!(df, [:y, :y_lag] .=> (x -> log.(x .+ 1)) .=> [:y, :y_lag])
    select!(df, Not(reference_country))

    df
end

function calc_default_stats(df_model::DataFrame)
    stat_dict = @pipe df_model |>
        groupby(_, [:country, :quality]) |>
        combine(
            _, 
            :y => mean,
            :y => std
        ) |>
        Dict(tuple.(_.country, _.quality) .=> tuple.(_.y_mean, _.y_std))

    stats = DataFrame([stat_dict[(c, q)] for (c, q) in zip(df_model.country, df_model.quality)])

    stats[:, 1], stats[:, 2]
end