using CSV, DataFrames, Pipe
using Statistics, Distributions, Random

Random.seed!(1234)

df_transformed = CSV.File("data/transformed_data.csv") |> DataFrame |> dropmissing

#### OLS model

# using GLM

# df_lm = @pipe df_transformed |>
#     select(_, Not([:country, :year])) |>
#     dropmissing

# lm(Term(:y) ~ sum(Term.(Symbol.(names(df_lm[:, Not(:y)])))), df_lm)

#### Bayesian model

### Data

## We're going to prepare dict with descriptive stats for easy replacing missing values by random variables

df_popular_summary = @pipe df_transformed |>
    dropmissing |>
    groupby(_, [:country, :quality]) |>
    combine(
        _, 
        :y => mean,
        :y => std
    ) |>
    sort(_, :y_mean)

stat_dict = Dict(
    tuple.(df_popular_summary.country, df_popular_summary.quality) .=> 
    tuple.(df_popular_summary.y_mean, df_popular_summary.y_std)
)

##

y = df_transformed.y
y_lag = df_transformed.y_lag
X = @pipe df_transformed |>
    select(_, Not([:country, :year, :y, :y_lag])) |>
    Matrix

length(y_lag)
size(X, 1)

### Constants

const default_mus = [stat_dict[(c, q)][1] for (c, q) in zip(df_transformed.country, df_transformed.quality)]
const default_sigmas = [stat_dict[(c, q)][2] for (c, q) in zip(df_transformed.country, df_transformed.quality)]

country_mean_prior_coeffs = Dict(
    "quality" => -0.4,
    # "Premium" => -0.2,
    # "Super Premium" => -0.5,

    "Belgium and Luxembourg" => 0.2,
    "United States" => 0.3,    # United States
    "Ireland" => 0.5,    # Ireland
    "France" => 0.2,    # France
    # "South Africa" => 0.5,    # South Africa
    # "Russia" => 0.2,    # Russia
    "Germany" => 0.5,    # Germany
    "United Kingdom" => 0.3,    # United Kingdom
    # "Czech Republic" => 0.5,    # Czech Republic
    "Canada" => 0.3,    # Canada
    # "Bulgaria" => 0.2,    # Bulgaria
    # "Australia" => 0.5,    # Australia
    "Sweden" => 0.2,    # Sweden
    # "Poland" => 0.5,    # Poland
    "Portugal" => 0.2,    # Portugal
    # "Slovakia" => 0.5,    # Slovakia
    "Denmark" => 0.2,    # Denmark
    "Netherlands" => 0.2,    # Netherlands
    # "Lativia" => 0.5,    # Lativia
    "Greece" => 0.1,    # Greece
    # "Ukraine" => 0.2,    # Ukraine
    "Spain" => 0.1,    # Spain
    # "Lithuania" => 0.5,    # Lithuania
    # "Norway" => 0.5,    # Norway
    # "Japan" => 0.2,    # Japan
    # "New Zealand" => 0.5,    # New Zealand
    # "Italy" => 0.5,    # Italy
    "Finland" => 0.2,    # Finland
    "Austria" => 0.2,    # Austria
    # "Estonia" => 0.5,    # Estonia
    # "Argentina" => 0.2,    # Argentina
    # "Romania" => 0.2,    # Romania
    # "Hungary" => 0.5,    # Hungary
    # "Switzerland" => 0.5,    # Switzerland
    # "Thailand" => 0.2,    # Thailand
)
const X_mu_prior = [
    country_mean_prior_coeffs[c]
    for c in vcat("quality", unique(df_transformed.country))
]
const X_sigma_prior = 0.2

# We assume that y[t] is strongly correlated with y[t-1], let's say 0.8, and due to log-log relation we can put this as exptected value of suitable coeff
const lag_mu_prior = 0.8
const lag_sigma_prior = 0.2

# We know very few about constant so we're setting high variance

const intercept_mu_prior = 3
const intercept_sigma_prior = 25

### Param simplification

# const coeff_mu_prior = vcat(intercept_mu_prior, y_lag_mu_prior, binary_mu_prior)
# const coeff_sigma_prior = vcat(intercept_sigma_prior, y_lag_sigma_prior, fill(binary_sigma_prior, size(X, 2)))
const missing_lag_idx =  filter(i -> ismissing(y_lag[i]), eachindex(y_lag)) #collect(eachindex(y_lag))[ismissing.(y_lag)]
# const mus_for_missing_obs = default_mus[is_y_lag_missing]
# const sigmas_for_missing_obs = default_sigmas[is_y_lag_missing]

size(X, 2)
length(X_mu_prior)

#### Actual bayesian model

using Turing, MCMCChains, StatsPlots
using LazyArrays

@model function mvar_reg1(y, y_lag, X)
    σ² ~ InverseGamma(1, 1)

    α ~ Normal(intercept_mu_prior, intercept_sigma_prior)
    
    β₁ ~ Normal(lag_mu_prior, lag_sigma_prior)

    # β ~ filldist(Normal(), length(X_mu_prior))
    # β = (β .* X_sigma_prior) .+ X_mu_prior

    # β ~ MvNormal(X_mu_prior, X_sigma_prior)

    β ~ arraydist(LazyArray(@~ Normal.(X_mu_prior, X_sigma_prior)))

    # for i in missing_lag_idx
    #     y_lag[i] ~ Normal(default_mus[i], default_sigmas[i])
    # end

    y_lag ~ arraydist(LazyArray(@~ Normal.(default_mus, default_sigmas)))

    μ = α .+ y_lag .* β₁ .+ X * β
    y ~ MvNormal(μ, sqrt(σ²))
end

# @model function mvar_reg2(y, y_lag, X)
#     v = length(y) - 2

#     σ ~ Exponential(1 / std(y))
#     α ~ TDist(v)
#     α = α * intercept_sigma_prior + intercept_mu_prior
    
#     β₁ ~ TDist(v)
#     β₁ = β₁ * lag_sigma_prior + lag_mu_prior

#     β ~ filldist(TDist(v), length(X_mu_prior))
#     β = β .* X_sigma_prior .+ X_mu_prior

#     y_lag ~ arraydist(LazyArray(@~ Normal.(default_mus, default_sigmas)))

#     μ = α .+ y_lag .* β₁ .+ X * β
#     y ~ MvNormal(μ, σ)

#     return y
# end

model = mvar_reg1(y, y_lag, X)

alg = NUTS(1000, 0.65)
prior_chain = sample(model, Prior(), 1000)
chain = @time sample(model, alg, MCMCThreads(), 300, 8)

plot(chain, seriestype = :mixeddensity, dpi=300)
plot(chain, seriestype = (:meanplot, :histogram), colordim = :parameter, dpi=300)

savefig("data/chains_plots.png")

# using JLD2
# jldsave("data/sampled_chain.jld2"; chain)
1+2