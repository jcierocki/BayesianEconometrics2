using CSV, DataFrames, Pipe
using Statistics, Distributions, Random

Random.seed!(1234)

df_transformed = CSV.File("data/transformed_data.csv") |> DataFrame

#### OLS model

using GLM

df_lm = @pipe df_transformed |>
    select(_, Not([:quality, :country, :year])) |>
    dropmissing

lm(Term(:y) ~ sum(Term.(Symbol.(names(df_lm[:, Not(:y)])))), df_lm)

#### Bayesian model

### Data

## We're going to prepare dict with descriptive stats for easy replacing missing values by random variables

df_popular_summary = @pipe df_transformed |>
    dropmissing |>
    groupby(_, [:country, :quality]) |>
    combine(
        _, 
        :y => mean,
        :y => (x -> sqrt(var(x))) => :y_sd
    ) |>
    sort(_, :y_mean)

stat_dict = Dict(
    tuple.(df_popular_summary.country, df_popular_summary.quality) .=> 
    tuple.(df_popular_summary.y_mean, df_popular_summary.y_sd)
)

##

y = df_transformed.y
y_lag = df_transformed.y_lag
X = @pipe df_transformed |>
    select(_, Not([:quality, :country, :year, :y, :y_lag])) |>
    Matrix

length(y_lag)
size(X, 1)

### Constants

const default_mus = [stat_dict[(c, q)][1] for (c, q) in zip(df_transformed.country, df_transformed.quality)]
const default_sigmas = [stat_dict[(c, q)][2] for (c, q) in zip(df_transformed.country, df_transformed.quality)]

binary_mu_prior = [
    -0.2,   # Premium
    -0.5,   # Super Premium

    0.8,    # United States
    1.0,    # Ireland
    0.5,    # France
    0.5,    # South Africa
    0.2,    # Russia
    0.5,    # Germany
    0.8,    # United Kingdom
    0.5,    # Czech Republic
    0.8,    # Canada
    0.2,    # Bulgaria
    0.5,    # Australia
    0.5,    # Sweden
    0.5,    # Poland
    0.5,    # Portugal
    0.5,    # Slovakia
    0.5,    # Denmark
    0.5,    # Netherlands
    0.5,    # Lativia
    0.5,    # Greece
    0.2,    # Ukraine
    0.5,    # Spain
    0.5,    # Lithuania
    0.5,    # Norway
    0.2,    # Japan
    0.5,    # New Zealand
    0.5,    # Italy
    0.5,    # Finland
    0.5,    # Austria
    0.5,    # Estonia
    0.2,    # Argentina
    0.2,    # Romania
    0.5,    # Hungary
    0.5,    # Switzerland
    0.2,    # Thailand
]
binary_sigma_prior = 0.5

# We assume that y[t] is strongly correlated with y[t-1], let's say 0.8, and due to log-log relation we can put this as exptected value of suitable coeff
y_lag_mu_prior = 0.8
y_lag_sigma_prior = 0.15

# We know very few about constant so we're setting high variance

intercept_mu_prior = 3
intercept_sigma_prior = 25

### Param simplification

const coeff_mu_prior = vcat(intercept_mu_prior, y_lag_mu_prior, binary_mu_prior)
const coeff_sigma_prior = vcat(intercept_sigma_prior, y_lag_sigma_prior, fill(binary_sigma_prior, size(X, 2)))
const is_y_lag_missing =  filter(i -> ismissing(y_lag[i]), eachindex(y_lag)) #collect(eachindex(y_lag))[ismissing.(y_lag)]
const mus_for_missing_obs = default_mus[is_y_lag_missing]
const sigmas_for_missing_obs = default_sigmas[is_y_lag_missing]

#### Actual bayesian model

using Turing, MCMCChains, StatsPlots
using DynamicHMC
using Tracker

@model function whiskey_consumption_regression(y, y_lag, X)
    σ² ~ InverseGamma(1, 1)
    # α ~ Normal(mean(y), 100)
    
    # β ~ MvNormal(coeff_mu_prior, coeff_sigma_prior)
    β ~ filldist(Normal(), length(coeff_mu_prior))
    β = β .* coeff_sigma_prior .+ coeff_mu_prior

    # y_lag[is_y_lag_missing] ~ MvNormal(mus_for_missing_obs, sigmas_for_missing_obs)
    for i in is_y_lag_missing
        y_lag[i] ~ Normal(default_mus[i], default_sigmas[i])
    end

    μ = β[1] .+ y_lag .* β[2] .+ X * β[3:end]
    # μ = hcat(fill(1, 1684), y_lag, X) * β
    y ~ MvNormal(μ, sqrt(σ²))
end

model = whiskey_consumption_regression(y, y_lag, X)
# alg = Gibbs(
#     NUTS{Turing.TrackerAD}(1000, 0.65, :β),
#     NUTS{Turing.ForwardDiffAD{1}}(1000, 0.65, :y_lag),
#     NUTS{Turing.TrackerAD}(1000, 0.65, :y),
# )
# alg = NUTS{Turing.TrackerAD}(1000, 0.65)
alg = NUTS(1000, 0.65)
# alg = DynamicNUTS()
chain = @time sample(model, alg, MCMCThreads(), 1000, 16)

plot(chain, seriestype = :mixeddensity, colordim = :parameter, size=(1200, 19200))

savefig("data/chains_plots.png")

# using JLD2
# jldsave("data/sampled_chain.jld2"; chain)