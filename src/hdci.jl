using KernelDensity, Plots; gr()
	
struct HDCI{T<:AbstractFloat}
    p::T
    start::Int64
    stop::Int64
    x::Vector{T}
    dens::Vector{T}
end

function HDCI(s::Vector{T}, p::T)::HDCI{T} where T<:AbstractFloat
    s = sort(s)
    dens = kde(s)

    xs = collect(dens.x)
    emp_pdf = dens.density ./ sum(dens.density)
    emp_cdf = accumulate(+, emp_pdf)

    start = 1
    stop = findfirst(x -> x >= p, emp_cdf)
    curr_prob = emp_cdf[stop]
    min_window_size = stop
    best_window = (start, stop)

    while stop < length(emp_pdf)
        curr_prob -= emp_pdf[start]
        start += 1

        while stop < length(emp_pdf) && curr_prob < p
            stop += 1
            curr_prob += emp_pdf[stop]
        end

        curr_window_size = stop - start
        if curr_window_size < min_window_size
            min_window_size = curr_window_size
            best_window = (start, stop)
        end
    end

    start, stop = best_window
    return HDCI(p, start, stop, xs,	emp_pdf)
end

function plot(hdci::HDCI{T}) where T<:AbstractFloat
    group = vcat(
        repeat(["tail"], hdci.start),
        repeat(["body"], hdci.stop - hdci.start),
        repeat(["tail"], length(hdci.x) - hdci.stop)
    )
    
    bar(hdci.x, hdci.dens, group = group, linecolor = :match, legend = false, title = "$(hdci.p)% confidence interfal")
end