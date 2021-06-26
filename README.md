# International differences in irish whiskey sales

This repo contains project prepared for the Bayesian Econometrics master classes at SGH Warsaw School of Economics ([link](https://web.sgh.waw.pl/~atoroj/)). The classes were orginally tought using _R_ and _RStan_ but I was allowed to prepare the 2nd assignment in Julia and [Turing.jl](https://turing.ml/stable/). I have choosen Julia as it provides rapidly developing packages ecosystem for bayesian modelling and enables preparing complete solution in one language, without splitting it into i.e. R and Stan.

This particular analysis focuses on providing an bayesian regression model with specified _a priori_ expecations which explains differences in the irish whiskey sales dynamics across the countries. I used the data provided by the [_Irish Food Board_](https://www.bordbia.ie) in 2018, which consist of sales records from 1995 to 2017 for multiple countries and 3 quality categories. In this particular research I focused on EU14 (before Brexit) countries plus US and Canda.

For pre-rendered static .html reports (in Polish) check:
- [report_pl.html](report_pl.html)
- [report_with_codes_pl.html](report_with_codes_pl.html)

For source _Pluto.jl_ notebook check:
- [notebook.jl](notebook.jl)

TODO's:
- english translation of the report,
- forecasts and out-of-sample evaluation,
- additional analysis using Variational Inference.
