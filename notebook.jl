### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 0635c68d-acea-49fe-bf19-6d2913f7ca59
begin
	using CSV, DataFrames, Pipe
	
	df = CSV.File("data/transformed_data.csv") |> DataFrame
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
## Opis zbioru danych

W dalszej analizie zostaną wykorzystane dane zebrane przez [_The ISWR_](https://twitter.com/TheIWSR), a opublikowane przez [_Irish Food Board_](https://www.bordbia.ie), z okazji obchodzów Dnia Św. Patryka 2018, na potrzeby prostego konkursu wizualizacyjnego. Obecnie dostęp do danych można uzyskać za pomocą portalu [_data.world_](https://data.world/makeovermonday/2018w11-growth-in-irish-whiskey-sales/workspace/project-summary?agentid=makeovermonday&datasetid=2018w11-growth-in-irish-whiskey-sales).

Surowe dane zawierają 5 kolumn po 4131 wierszy i mają charakter makro-panelu w formacie wzdłużnym: każdy wiersz zawiera informacje o skumulowanej sprzedaży whiskey irlandzkiej dla konkretnego kraju, roku oraz klasy jakości (standardowa, premium lub "super premium").

W celu ograniczenia liczby anomalii, braków danych oraz redukcji wymiaru (po konwersji zmiennych kategoryzowanych na binarne) zbiór danych ograniczono do 14 krajów EU14 (na rok 2018) oraz państw anglosaskich Ameryki Płn.: USA i Kanady. Dane o oryginalnym wymiarze również zostały przetestowane, ale ich użycie wiązało się z poważnymi trudności w zakresie obliczeń numerycznych, które w optymistycznym przypadku sprowadzały się do kilkudziesięciominutowych czasowych pracy algorytmu NUTS, a uzyskane wyniki, w szczególności wykresy, były bardzo trudne do analizy.

Dane oryginalne były wyrażone w wartościach bezwzględnych i w celu ich przeskalowania do postaci per capita (skumulowana sprzedaż roczna na 1 mln mieszkańców: `popularity`) zostały wykorzystane dane demograficzne publikowane przez _Bank Światowy_. Dodano ponadto jej opóźnienie 1 rzędu: `popularity_lag`, które będzie dalej wykorzystywane jako zmienna objaśniająca.

Proces wstępnej obróbki danych omówionych powyżej został zaimplementowany w pliku `preproc.jl`. W wyniku jego zastosowania otrzymaliśmy tabelę w postaci:
"""

# ╔═╡ ea6add69-0466-4fbc-9d29-d37fe8c2afa1
md"""
Zmienna `quality`, wyrażająca klasę jakości trunku, po początkowych próbach z konwersją na binarne, została potraktowana jako zmienna liczbowa z wartościami odpowiednio:
- Standard: -1,
- Premium: 0,
- Super Premium: 1,
co było możliwe z racji jej uporządkowania i generowania monotonicznej zależności ze zmienną objaśnianą.
"""

# ╔═╡ 3433bc17-fc1d-428f-a62f-d731c25540b6


# ╔═╡ 20caf9c7-a33e-4b05-8675-8d03124eadc3


# ╔═╡ a22bfe62-2b6d-423b-aa75-73024ebf59ef


# ╔═╡ 05bfa48b-117d-44ed-bdab-4784421cded9


# ╔═╡ db5fb29b-a421-4ec9-8389-1e6b63620aef


# ╔═╡ a294d298-8341-4ccd-acd8-8018b4651756


# ╔═╡ e9aea77d-2a67-4785-9ab1-9e5e4b2133ad


# ╔═╡ c163823e-6b64-40ae-a0e4-a70c098bca2d


# ╔═╡ a53fd440-2e99-4679-a8e5-9e2b9a7dce75


# ╔═╡ 37a2021a-3992-4f74-8709-048074385a52


# ╔═╡ 35459938-4618-4bd4-a08a-847e98ae155b


# ╔═╡ f0bf74a8-5183-4b4c-b3d9-e34d45bc7d1e


# ╔═╡ 30f07b05-ae6b-44dd-930a-d3c9f3f87e4d


# ╔═╡ ed513c45-110d-40bd-9918-0e9b7e7df6be


# ╔═╡ 9661eaaa-9306-48bd-bc8a-0196a8cd4be9


# ╔═╡ 11310ca9-f533-4c26-ba5c-fc8db7c21d1e


# ╔═╡ 9d9fccfb-2ade-48a5-96cd-38a37f1c1cd5


# ╔═╡ a7679cf6-0e25-4e0a-ac16-fbf35eb5b9ce


# ╔═╡ fc82c472-067f-46df-9ca8-ee7cacc77d27


# ╔═╡ 285568be-c244-45d7-bdf1-83c7f93d04a9


# ╔═╡ e2520e8e-46e5-41e4-843a-9acbc46ba722


# ╔═╡ ac772dc6-1f0f-4c4f-9e08-8174f9fd98a0


# ╔═╡ d12a260f-e84c-4238-bdfd-559c43efa388


# ╔═╡ 3159c378-8dce-4781-9989-1d267f38612b


# ╔═╡ 7e770ecc-2172-4ea5-82b2-75cd037098ed


# ╔═╡ 830d32a3-30e6-4cfa-898b-1536de73b7f0


# ╔═╡ 5b26c0a7-abc4-4725-b8aa-1a0a05a2c591


# ╔═╡ 1a12c93d-6fcd-47ca-925f-7c975e1d1160


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
# ╠═ea6add69-0466-4fbc-9d29-d37fe8c2afa1
# ╠═3433bc17-fc1d-428f-a62f-d731c25540b6
# ╠═20caf9c7-a33e-4b05-8675-8d03124eadc3
# ╠═a22bfe62-2b6d-423b-aa75-73024ebf59ef
# ╠═05bfa48b-117d-44ed-bdab-4784421cded9
# ╠═db5fb29b-a421-4ec9-8389-1e6b63620aef
# ╠═a294d298-8341-4ccd-acd8-8018b4651756
# ╠═e9aea77d-2a67-4785-9ab1-9e5e4b2133ad
# ╠═c163823e-6b64-40ae-a0e4-a70c098bca2d
# ╠═a53fd440-2e99-4679-a8e5-9e2b9a7dce75
# ╠═37a2021a-3992-4f74-8709-048074385a52
# ╠═35459938-4618-4bd4-a08a-847e98ae155b
# ╠═f0bf74a8-5183-4b4c-b3d9-e34d45bc7d1e
# ╠═30f07b05-ae6b-44dd-930a-d3c9f3f87e4d
# ╠═ed513c45-110d-40bd-9918-0e9b7e7df6be
# ╠═9661eaaa-9306-48bd-bc8a-0196a8cd4be9
# ╠═11310ca9-f533-4c26-ba5c-fc8db7c21d1e
# ╠═9d9fccfb-2ade-48a5-96cd-38a37f1c1cd5
# ╠═a7679cf6-0e25-4e0a-ac16-fbf35eb5b9ce
# ╠═fc82c472-067f-46df-9ca8-ee7cacc77d27
# ╠═285568be-c244-45d7-bdf1-83c7f93d04a9
# ╠═e2520e8e-46e5-41e4-843a-9acbc46ba722
# ╠═ac772dc6-1f0f-4c4f-9e08-8174f9fd98a0
# ╠═d12a260f-e84c-4238-bdfd-559c43efa388
# ╠═3159c378-8dce-4781-9989-1d267f38612b
# ╠═7e770ecc-2172-4ea5-82b2-75cd037098ed
# ╠═830d32a3-30e6-4cfa-898b-1536de73b7f0
# ╠═5b26c0a7-abc4-4725-b8aa-1a0a05a2c591
# ╠═1a12c93d-6fcd-47ca-925f-7c975e1d1160
# ╠═413ee453-6a1f-4f8e-8317-bfb76166c4f2
# ╠═98e9f972-13bb-4e59-b8e4-520af64f6d1c
# ╠═54e4a046-d4de-42e1-83f5-d27a2ca68f70
# ╠═1bb21042-575b-4b6d-bbcd-d93a527a5eda
# ╠═63168d61-1e3b-4fd0-95ef-5242ec6016a5
# ╠═d5506de4-cc3c-11eb-1c78-b72379ea4d38
