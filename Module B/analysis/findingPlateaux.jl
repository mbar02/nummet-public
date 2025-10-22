using Printf
#using Scanf
using Plots
using LaTeXStrings
using ProgressBars
using Statistics
using Smoothers
using JSON3
using ArgParse

function parse_arguments()
  s = ArgParse.ArgParseSettings()

  @add_arg_table! s begin
    "descriptor"
      help     = "The JSON descriptor file of the simulation."
      required = true
    "--nogui"
      help     = "Flag to disable GUI for the plots. Remember to use export GKSwstype=\"nul\" with this option."
      action   = :store_true
  end
  return parse_args(s)

end

function find_k(k, s; points::Int=5)
  mean_s = sma(s, points, false)
  mean_k = sma(k, points, false)

  # compute the mean of the next points
  sum_next_s = zero(mean_k)

  sum_next_s[end] = mean_s[end]
  for i = (length(mean_s)-1):-1:1
    sum_next_s[i] = sum_next_s[i+1] + mean_s[i]
  end

  k   = NaN
  ms  = NaN
  # find k if mean_s[k] >= mean_next_s[k] * 0.90
  for i = 1:( length(mean_s)*7÷8 )
    if mean_s[i] >= sum_next_s[i+1] / ( length(mean_s) - i ) * 0.90
      k   = mean_k[i]
      ms  = mean_s[i]
      break
    end
  end

  return (ms, k)
end

function (@main)(args)
  parsed_args = parse_arguments()

  nogui::Bool      = parsed_args["nogui"]
  sim_desc::String = parsed_args["descriptor"]
  
  if nogui && get(ENV,"GKSwstype","") != "nul"
    println("Remember to use export GKSwstype=\"nul\" with this option!")
    exit(136)
  end

  desc_dict::Dict{Symbol,Any} = Dict{Symbol,Any}()
  
  try
    desc_dict=JSON3.read(sim_desc; allow_inf=true)
  catch _
    println(stderr, "Unable to read the descriptor file. The file is either missing or corrupted: " * sim_desc)
    return 136
  end

  namefig_prefix::String = joinpath(desc_dict[:folder],"blocking-")
  input_file::String     = joinpath(desc_dict[:folder],"data.csv")
  output_file::String    = joinpath(desc_dict[:folder],"data-blocked.csv")


  if !isfile(input_file)
    println("Unable to read the simulation data file. The file is either missing or corrupted: " * input_file)
    throw(ArgumentError("File $input_file does not exist."))
    return 136
  end

  N::Int = 100
  lines = desc_dict[:samples]

  k = unique(floor.(Int, logrange(1, lines / 10, N)))
  N = length(k)
  n_measures = zeros(Int, N)

  nLags   = length(desc_dict[:lags]) 
  maxXpow = 4        # number of couple of observables
  M = maxXpow + nLags * ( maxXpow * (maxXpow+1) ÷ 2 ) #total number of observables == columns in csv file

  obss_labels = Vector{String}(undef, M)
  pos = 1
  for i = 1:maxXpow
    obss_labels[pos] = "x{$i}"
    pos+=1
  end
  for lag = 1:nLags
    for i=1:maxXpow
      for j = i:maxXpow
        obss_labels[pos] = "C{$i,$j}($lag)"
        pos+=1
      end
    end
  end

  observables     = Vector{Function}(undef, M)
  for i =1:M
    observables[i] = x -> x[i+1] 
  end
  #format_string = Scanf.Format( ("%lf, " ^ M) * "%lf\n" )
  #format_type = Vector{Type}(undef, M+1)
  #fill!(format_type, Float64)

  sums    = zeros(Float64, M, N)
  sums2   = zeros(Float64, M, N)
  last    = zeros(Float64, M, N)
  sigma2  = zeros(Float64, M, N)
 
  lengths = zeros(Int, N)
  thermalization = lines ÷ 10

  if !nogui
    pbar = ProgressBar(total=lines-1-thermalization, printing_delay=0.5)
  end

  open(input_file) do file
    for i = 1:(thermalization+1)
      readline(file)
    end
    for line in eachline(file)
      if !nogui
        update(pbar)
      end
      new_line = parse.(Float64, split(line, ','))
      for i = 1:N
        for j = 1:M
          last[j, i] += observables[j](new_line)
        end
        lengths[i] += 1
        if (lengths[i] >= k[i])
          for j = 1:M
            sums[j, i] += last[j, i]
            sums2[j, i] += last[j, i]^2

            last[j, i] = 0
          end
          lengths[i] = 0
          n_measures[i] += 1
        end
      end
    end
  end

  ks = Vector{Float64}(undef, M)

  colors = palette(:navia,length(obss_labels)+2)[2:end-1]

  p = plot([],[],label=nothing)

  pp = Vector{Plots.Plot{Plots.GRBackend}}(undef, M)

  for j in 1:M
    @. sigma2[j, :] = ( sums2[j, :] - sums[j, :]^2 / n_measures) / (n_measures * (n_measures - 1) * k^2)
    # Reduced sigma2 (Not used because <x>, or odd powers, vanishes!)
    # @. sigma2[j, :] = ( sums2[j, :] - sums[j, :]^2 / n_measures) / (n_measures * (n_measures - 1) * k^2) / ( sums[j,1] / n_measures[1] )^2
    ms, ks[j] = find_k(k, sigma2[j, :])
    desc_dict[Symbol(obss_labels[j] * "_k")] = ks[j] * 10 # factor 10 for safety
    pp[j] = plot(k, sigma2[j, :], label=obss_labels[j], color=colors[j])
    if !isnan(ks[j])
      plot!(pp[j], [ks[j]], [ms], color=colors[j], seriestype=:scatter, markershape=:diamond, markersize=6, label=nothing)
    end
  end

  open(sim_desc, "w") do io
    JSON3.pretty(io, desc_dict; allow_inf=true)
  end

  for (idx, p) in enumerate(pp)
    plot!(
      p,
      xscale=:log10, yscale=:log10,
      minorgrid=true,
      legend=false,
  #    fontfamily="Times New Roman",
      size=(430, 300),
      dpi=500,
      tickfontsize=8,
      legendfontsize=10,
      labelfontsize=10,
      xlabel="Blocking size", ylabel=L"\sigma^2_X / \langle X \rangle^2",
    )

    savefig(p, namefig_prefix * string(idx) * ".svg")

    if (!nogui)
      display(p)
      gui()
      readline()
    end
  end

  for j in 1:M
    if isnan(ks[j])
      println("Unable to find a plateau for observable $(obss_labels[j]).")
      exit(136)
    end
  end

  return 0
end
