using CSV, DataFrames, Pipe
using Statistics, Distributions, Random

Random.seed!(1234)

include("src/preproc_api.jl")

df_transformed = @pipe CSV.File("data/transformed_data.csv") |> DataFrame |> prepare_model_df(_, :Italy)

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
    # "Greece" => 0.1,    # Greece
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

cols = filter(c -> c != "Italy", vcat("quality", unique(df_transformed.country)))

const X_mu_prior = [
    country_mean_prior_coeffs[c]
    for c in cols
]
const X_sigma_prior = 0.2

# We assume that y[t] is strongly correlated with y[t-1], let's say 0.8, and due to log-log relation we can put this as exptected value of suitable coeff
const lag_mu_prior = 0.8
const lag_sigma_prior = 0.2

# We know very few about constant so we're setting high variance

const intercept_mu_prior = 3
const intercept_sigma_prior = 25

size(X, 2)
length(X_mu_prior)

#### Actual bayesian model

using Turing, MCMCChains, StatsPlots
using LazyArrays

@model function mvar_reg1(y, y_lag, X)
    ???? ~ InverseGamma(1, 1)

    ?? ~ Normal(intercept_mu_prior, intercept_sigma_prior)
    
    ????? ~ Normal(lag_mu_prior, lag_sigma_prior)

    # ?? ~ MvNormal(X_mu_prior, X_sigma_prior)
    ?? ~ arraydist(LazyArray(@~ Normal.(X_mu_prior, X_sigma_prior)))

    # y_lag ~ arraydist(LazyArray(@~ Normal.(default_mus, default_sigmas)))
    for i in eachindex(y_lag)
        y_lag[i] ~ Normal(default_mus[i], default_sigmas[i])
    end
    

    ?? = ?? .+ y_lag .* ????? .+ X * ??
    y ~ MvNormal(??, sqrt(????))
end

# @model function mvar_reg2(y, y_lag, X)
#     v = length(y) - 2

#     ?? ~ Exponential(1 / std(y))
#     ?? ~ TDist(v)
#     ?? = ?? * intercept_sigma_prior + intercept_mu_prior
    
#     ????? ~ TDist(v)
#     ????? = ????? * lag_sigma_prior + lag_mu_prior

#     ?? ~ filldist(TDist(v), length(X_mu_prior))
#     ?? = ?? .* X_sigma_prior .+ X_mu_prior

#     y_lag ~ arraydist(LazyArray(@~ Normal.(default_mus, default_sigmas)))

#     ?? = ?? .+ y_lag .* ????? .+ X * ??
#     y ~ MvNormal(??, ??)

#     return y
# end

model = mvar_reg1(y, y_lag, X)
alg = NUTS(1000, 0.65)
prior_chain = sample(model, Prior(), 1000)
chain = @time sample(model, alg, MCMCThreads(), 1000, 4)

chain_sub = chain[vcat("??", "?????", ["??[$idx]" for idx in 1:15])]

plot(chain_sub, seriestype = :mixeddensity, dpi=300)
plot(chain_sub, seriestype = (:meanplot, :histogram), colordim = :parameter, dpi=300)

summarize(chain)

savefig("data/chains_plots.png")


