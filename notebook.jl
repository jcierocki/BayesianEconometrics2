### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 0635c68d-acea-49fe-bf19-6d2913f7ca59
begin
	using CSV, DataFrames, Pipe
	
	df = CSV.File("data/transformed_data.csv") |> DataFrame
end

# ╔═╡ c0cdb478-c615-49a0-9cb5-e30c2cef1834
begin
	using Plots; gr()
	theme(:default)
	
	@pipe df |>
		groupby(_, :year) |>
		combine(_, :popularity => (x->sum(x; init = 0)) => :sum) |>
		plot(
			_.year,
			_.sum, 
			title = "Skumulowana sprzedaż irlandzkiej whiskey na świecie",
			legend = false,
			xlabel = "rok",
			size = (640, 320)
		)
end

# ╔═╡ 37a2021a-3992-4f74-8709-048074385a52
begin
	include("src/preproc_api.jl")
	
	df_model = prepare_model_df(df, :Italy)
	const default_mu, default_sigma = calc_default_stats(df_model)
	
	df_model
end

# ╔═╡ 285568be-c244-45d7-bdf1-83c7f93d04a9
begin
	using GLM

	df_lm = @pipe df_model |>
		select(_, Not([:country, :year])) |>
		dropmissing

	lm(Term(:y) ~ sum(Term.(Symbol.(names(df_lm[:, Not(:y)])))), df_lm)
end

# ╔═╡ 31ade8be-846f-4597-8bac-76156f2b0069
html"""
<center>
<h1>Międzynarodowe zróżnicowanie popularności whiskey irlandzkiej - analiza z wykorzystaniem metod bayesowskich</h1>
Jakub Cierocki<br>
Ekonometria bayesowska, MIESI, SGH<br>
2021, 13 czerwca

</center>
"""

# ╔═╡ 28067a58-cdf4-4051-acbe-a55d8fb07197
md"""
## Wprowadzenie

W niniejszej pracy zostanie przeanalizowana sprzedaż whiskey irlandzkiej per capita w latach 1990-2016 w krajach EU14 (sprzed Brexitu), USA oraz Kanadzie. Jej konsumpcja ustawicznie rośnie na przestrzeni ostatnich lat, jest jednak bardzo zróżnicowana między krajami i ze względu na klasy jakości. Celem tej pracy będzie w tej sytuacji pogłębienie oczywistych wniosków wynikających z opisowej analizy danych oraz ich sformalizowanie w postaci modelu matematycznego. Podejście bayesowskie jest w tym przypadku szczególnie obiecujące ze względu na liczne braki danych ładność elicytacji a priori. 

"""

# ╔═╡ 795dc75f-2ad7-4f26-ba49-a41b77aa5228
md"""
## Zbiór danych

W dalszej analizie zostaną wykorzystane dane zebrane przez [_The ISWR_](https://twitter.com/TheIWSR), a opublikowane przez [_Irish Food Board_](https://www.bordbia.ie), z okazji obchodzów Dnia Św. Patryka 2018, na potrzeby prostego konkursu wizualizacyjnego. Obecnie dostęp do danych można uzyskać za pomocą portalu [_data.world_](https://data.world/makeovermonday/2018w11-growth-in-irish-whiskey-sales/workspace/project-summary?agentid=makeovermonday&datasetid=2018w11-growth-in-irish-whiskey-sales).

Surowe dane zawierają 5 kolumn po 4131 wierszy i mają charakter makro-panelu w formacie wzdłużnym: każdy wiersz zawiera informacje o skumulowanej sprzedaży whiskey irlandzkiej dla konkretnego kraju, roku oraz klasy jakości (standardowa, premium lub "super premium").

W celu ograniczenia liczby anomalii, braków danych oraz redukcji wymiaru (po konwersji zmiennych kategoryzowanych na binarne) zbiór danych ograniczono do 14 krajów EU14 (na rok 2018) oraz państw anglosaskich Ameryki Płn.: USA i Kanady. W dalszej kolejności ze zbioru usunięto dane o sprzedaży whiskey "Super Premium" z powodu małej liczby (2) obserwacji. Równocześnie okazało się koniecznym usunąć ze zbioru danych Grecję ze względu na brak jakichkolwiek danych dla klas "Premium" i "Super Premium". Dane o oryginalnym wymiarze również zostały przetestowane, ale ich użycie wiązało się z poważnymi trudności w zakresie obliczeń numerycznych, które w optymistycznym przypadku sprowadzały się do kilkudziesięciominutowych czasowych pracy algorytmu NUTS, a uzyskane wyniki, w szczególności wykresy, były bardzo trudne do analizy.

Dane oryginalne były wyrażone w wartościach bezwzględnych i w celu ich przeskalowania do postaci per capita (skumulowana sprzedaż roczna na 1 mln mieszkańców: `popularity`) zostały wykorzystane dane demograficzne publikowane przez _Bank Światowy_. Dodano ponadto jej opóźnienie 1 rzędu: `popularity_lag`, które będzie dalej wykorzystywane jako zmienna objaśniająca.

Proces wstępnej obróbki danych omówionych powyżej został zaimplementowany w pliku `src/preproc.jl`. W wyniku jego zastosowania otrzymaliśmy tabelę w postaci:
"""

# ╔═╡ 4ddffff8-dc04-4d56-ab86-63f3ba2525ce
md"""
Zmienna objaśniana dla zredukowanego zbioru danych ma postać:
"""

# ╔═╡ 584ab811-460d-4a8c-96a4-003cc1415971
begin
	@pipe df |>
		histogram(
			_.popularity,
			title = "Gęstość empiryczna",
			legend = false,
			size = (640, 320)
		)
end

# ╔═╡ 03c5e8e8-267f-417b-ae9f-9ea6cd3f1339
begin
	@pipe df |>
		transform(_, :popularity => (x -> log.(x .+ 1.0)) => :log) |>
		histogram(
			_.log,
			title = "Gęstość empiryczna logarytmu + 1",
			legend = false,
			size = (640, 320)
		)
end

# ╔═╡ ea6add69-0466-4fbc-9d29-d37fe8c2afa1
md"""
Ze względu na bardzo silną prawostronną skośność zmiennej objaśnianej w dalszej analizie zostanie wykorzystany jej logarytm (analogicznie dla zmiennej opóźnionej), który w dalszym ciągu jednak jest daleki od rozkładu normalnego.

Zmienna `quality`, wyrażająca klasę jakości trunku, po początkowych próbach z konwersją na binarne, została potraktowana jako zmienna liczbowa z wartościami odpowiednio:
- Standard: -1,
- Premium: 0,
- Super Premium: 1,
co było możliwe z racji jej uporządkowania i generowania monotonicznej zależności ze zmienną objaśnianą.

Zmienna `country` została z kolei, zgodnie z wcześniejszymi zapowiedziami, przekonwertowana na zmienne binarne odpowiadające poszczególnym krajom, z wyłączeniem Włoch, które zostały pominięte w celu uniknięcia współliniwości. Wybór Włoch wynika z najniższego dla tego kraju średniego wskaźnika spożycia, który czyni je dobrym punktem odniesienia.
"""

# ╔═╡ fc82c472-067f-46df-9ca8-ee7cacc77d27
md"""
## Klasyczny model ekonometryczny

Wyestymujemy teraz klasyczny model OLS, który docelowo będzie stanowić punkt odniesienia dla dalszej analizy bayesowskiej.
"""

# ╔═╡ ac772dc6-1f0f-4c4f-9e08-8174f9fd98a0
md"""
## Model bayesowski

### Specyfikacja
``\newline``
``\begin{array}{lcl} X & - & \text{macierz zmiennych objaśniających} \\ K & - & \text{liczba obserwacji zawierających braki danych} \\ \sigma^2 & \sim & InvGamma(\underline{\alpha}_{\sigma^2}, \underline{\beta}_{\sigma^2}) \\ \alpha & \sim & \mathcal{N}(\underline{\mu}_\alpha,\underline{\sigma}_\alpha) \\ \mathbf{\beta} & \sim & \mathcal{N}(\underline{\mathbf{\mu}}_\beta,\underline{\mathbf{\sigma}}_\beta) \\ y_i & \overset{\forall_{k=1 \dots K}}{\sim} & \mathcal{N}(\hat{\mu}_{y_i},\hat{\sigma}_{y_i}) \\ \overline{\mathbf{y}} & \sim & \mathcal{N}(\alpha + \mathbf{X} \times \mathbf{\beta}, \sigma^2) 
\end{array}``
	
Opisany model posida łącznie 36 parametrów _a priori_, po 2 dla stałej i odchylenia, oraz po 2 dla każdego z 16 parametrów modelu.

### Elicytacja parametrów _a priori_

Przyjmijmy `` \underline{\alpha}_{\sigma^2}, \underline{\beta}_{\sigma^2} = 1 ``, daje to nam rozkład o długim ogonie i duże wariancji dobrze obrazujący naszą ograniczoną wiedzę na temat wariancji składnika losowego.

Podobne podejście zostosujmy do stałej: `` \underline{\mu}_\alpha = 3, \underline{\sigma}_\alpha = 25 ``, duża wariancja odpowiada ograniczonej wiedzy o rozkładzie parametru.

Zmienne objaśniające zawarte w `` \mathbf{X} `` można podzielić na 3 grupy:
- opóźnioną zmienną objaśnianą ``y_{t-1}`` (zlogarytmowaną liczbową)
- zmienną jakościową uporządkowaną ``quality``
- zmienne binarne krajów

W celu elicytacji parametrów rozkładów współczynników z wektora ``\mathbf{\beta}`` korzystamy z własności modelu log-liniowego, pozwalającej określić przybliżone interpretacje parametrów:
- zależność log-log: zmiana ``x`` o ``1\:\%`` powoduje zmianę ``y`` o około ``\beta \:\%`` ceteris paribus
- zależność log-raw: zmiana ``x`` o ``1`` powoduje zmianę ``y`` o około ``100 * \beta \:\%``

W naszym modelu mamy do czynienia ze (w sposób oczywisty) niestacjonarnym procesem autoregresyjnym, tj. sprzedaż w danym roku jest silnie powiązana z ubiegłoroczną i reprezentuje podobny rząd wielkości. Przyjmijmy w tej sytuacji ``\mu = 0.8`` co oznacza, zgodnie z wcześniej przedstawionymi regułami, że ``y`` wzrośnie średnio o ``0.8\:\%`` przy wzroście ``x`` o ``1\:\%`` oraz ``\sigma = 0.2`` co w świetle reguły 3 sigm oznacza, że ``75\:\%`` masy rozkładu parametru znajdzie się w przedziale ``0.6 1``.

Zdroworoządnowo, a w dodatku mając w papięci wcześniejszej analizy klasycznej, zakładamy, że "wzrost" jakoś trudnku powinien wpływać na spadek jego popularności. Przyjmijmy w tej sytuacji ``\mu = -0.4`` co odpowiada spadkowi sprzedaży o ``40\:\%`` przy przejściu do wyższej klasy jakości. Z racji, że wartości tej jesteśmy już mniej pewni, dobieramy do niej relatywnie większe ``\sigma = 0.2``.

W przypadku zmiennych binarnych, interpretacja w modelu ze zlogarytmowaną zmienną objaśnianą jest dość prosta, a mianowicie zmiana wartości z "NIE" na "TAK" skutkuje wzrostem wartości zmiennej objaśnianej o ``100 * \beta \:\%``. Nie mają dostępu do szczegółowych badań na poziomach krajowych wyróżnijmy 4 różne grupy krajówi odpowiadające im wartości oczekiwane rozkładów:

- macierzystą Irlandię: ``0.5``
- kraje anglosaskie: ``0.3``
- pozostałe kraje Europy Środokowo-Zachodniej i Północnej: ``0.2``
- kraje basenu Morza Śródziemnego (Hiszpania): : ``0.1``

Nie będąc w stanie wyznacznyć miarodajnie wartości dla każego kraju z osobna, ani nawet nie do końca dla grup, skupiliśmy się na wyodrębnieniu 4 grup możliwie różnych pod względami kultury, klimatu i zamożności, w sposób powiązany z konsumpcją droższych alkoholi wysokoprocentowych oraz przedstawieniu dysproporcji międzygrupowych. Zakładamy, że te rzeczywiste będą zbliżone do różnic między wartościami oczekiwanymi rozkładów _a priori_. W celu uwzględnienia przede wszystkim zmienności wewnątrzgrupowej przyjmujemy relatywnie duże ``\sigma = 0.2``.

Zaporponowana specyfikacja uwzględnia również imputację bayesowską brakujących wartości zmiennej zależnej, ale w tym celu zostaną wykorzystane rozkłady empiryczne momenty dla poszczególnych podgrup (kraj-klasa jakości).
"""

# ╔═╡ d12a260f-e84c-4238-bdfd-559c43efa388
begin
	const lag_mu_prior = 0.8
	const lag_sigma_prior = 0.2

	const intercept_mu_prior = 3
	const intercept_sigma_prior = 25
	
	country_mean_prior_coeffs = Dict(
		"quality" => -0.4,

		"Belgium and Luxembourg" => 0.2,
		"United States" => 0.3,
		"Ireland" => 0.5,
		"France" => 0.2,
		"Germany" => 0.5,
		"United Kingdom" => 0.3,
		"Canada" => 0.3,
		"Sweden" => 0.2,
		"Portugal" => 0.2,
		"Denmark" => 0.2,
		"Netherlands" => 0.2,
		"Spain" => 0.1,
		"Finland" => 0.2,
		"Austria" => 0.2,
	)

	cols = filter(c -> c != "Italy", vcat("quality", unique(df_model.country)))

	const X_mu_prior = [
		country_mean_prior_coeffs[c]
		for c in cols
	]
	const X_sigma_prior = 0.2
	
	println("Prior's loaded")
end

# ╔═╡ 1a12c93d-6fcd-47ca-925f-7c975e1d1160
begin
	using Turing, LazyArrays

	@model function mvar_reg1(y, y_lag, X)
		σ² ~ InverseGamma(1, 1)

		α ~ Normal(intercept_mu_prior, intercept_sigma_prior)

		β₁ ~ Normal(lag_mu_prior, lag_sigma_prior)
		β ~ arraydist(LazyArray(@~ Normal.(X_mu_prior, X_sigma_prior)))

		for i in eachindex(y_lag)
			y_lag[i] ~ Normal(default_mus[i], default_sigmas[i])
		end

		μ = α .+ y_lag .* β₁ .+ X * β
		y ~ MvNormal(μ, sqrt(σ²))
	end
end

# ╔═╡ 5b26c0a7-abc4-4725-b8aa-1a0a05a2c591
md"""
### Implmentacja modelu

Opisany wyżej model został zaimplementowany z użyciem pakietu _Turing.jl_, napisanego od zera w Julii subjęzyka probabilistycznego pozwalającego budować modele w tym samym języku co resztę analizy zachowując ponadto wydajność zbliżoną do _Stan_'a oraz liczne analogie w zakresie logicznej struktury kodu. Wśród funkcjonalności tego narzędzia, które zostaną mocniej wykorzystane w niniejszej analizie jest automatyczna detekcja brakujących wartości bez konieczności definiowania dodatkowej flagi i wyrażenia warunkowego jak w _Stan_'ie.

Kod samego prezentuje się następująco:
"""

# ╔═╡ 413ee453-6a1f-4f8e-8317-bfb76166c4f2


# ╔═╡ 98e9f972-13bb-4e59-b8e4-520af64f6d1c


# ╔═╡ 54e4a046-d4de-42e1-83f5-d27a2ca68f70


# ╔═╡ 1bb21042-575b-4b6d-bbcd-d93a527a5eda


# ╔═╡ 63168d61-1e3b-4fd0-95ef-5242ec6016a5


# ╔═╡ d5506de4-cc3c-11eb-1c78-b72379ea4d38


# ╔═╡ Cell order:
# ╟─31ade8be-846f-4597-8bac-76156f2b0069
# ╟─28067a58-cdf4-4051-acbe-a55d8fb07197
# ╟─795dc75f-2ad7-4f26-ba49-a41b77aa5228
# ╟─0635c68d-acea-49fe-bf19-6d2913f7ca59
# ╟─4ddffff8-dc04-4d56-ab86-63f3ba2525ce
# ╟─c0cdb478-c615-49a0-9cb5-e30c2cef1834
# ╟─584ab811-460d-4a8c-96a4-003cc1415971
# ╟─03c5e8e8-267f-417b-ae9f-9ea6cd3f1339
# ╟─ea6add69-0466-4fbc-9d29-d37fe8c2afa1
# ╠═37a2021a-3992-4f74-8709-048074385a52
# ╟─fc82c472-067f-46df-9ca8-ee7cacc77d27
# ╟─285568be-c244-45d7-bdf1-83c7f93d04a9
# ╟─ac772dc6-1f0f-4c4f-9e08-8174f9fd98a0
# ╟─d12a260f-e84c-4238-bdfd-559c43efa388
# ╟─5b26c0a7-abc4-4725-b8aa-1a0a05a2c591
# ╟─1a12c93d-6fcd-47ca-925f-7c975e1d1160
# ╠═413ee453-6a1f-4f8e-8317-bfb76166c4f2
# ╠═98e9f972-13bb-4e59-b8e4-520af64f6d1c
# ╠═54e4a046-d4de-42e1-83f5-d27a2ca68f70
# ╠═1bb21042-575b-4b6d-bbcd-d93a527a5eda
# ╠═63168d61-1e3b-4fd0-95ef-5242ec6016a5
# ╠═d5506de4-cc3c-11eb-1c78-b72379ea4d38
