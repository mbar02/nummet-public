using ArgParse
using DelimitedFiles
using JSON
using Statistics
using LinearAlgebra
using Plots
using LaTeXStrings
using Printf
using ProgressBars
using Base.Threads

# Graphical configuration
default(
  dpi = 500,
  size = (430, 350),
  titlefont = font("Times New Roman", 10),
  guidefont = font("Times New Roman", 8),
  tickfont = font("Times New Roman", 8),
  legendfontsize = 8,
  fontfamily = "Times New Roman"
)

# ACHTUNG! dx must have vanishing mean.
@inline function _ac(dx::Vector{Float64}, i::Int32, l::Int32, s::Float64)
  n = l - i
  return mean( dx[1:n] .* dx[1+i : i + n] ) / s
end

function autocorrelation(
  x::Vector{Float64};
  low_cutoff::Float64  = 0.1,
  high_cutoff::Float64 = 0.4
)
  dx = x .- mean(x)
  s  = var(x)
  l  = Int32(length(dx))

  lags = Int[]
  acs  = Float64[]

  lastk  = Int32(0)
  lastac = _ac(dx, lastk, l, s)
  nextk  = lastk + Int32(1)
  nextac = _ac(dx, nextk, l, s)

  # while over the high_cutoff, advance and ignore
  while nextac > high_cutoff
    lastk  = nextk;  lastac = nextac
    nextk  = lastk + Int32(1)
    nextac = _ac(dx, nextk, l, s)
  end

  # starting measurement (the last >= high_cutoff)
  push!(lags, lastk)
  push!(acs, lastac)

  # advance once and collect while above low_cutoff
  lastk = nextk;  lastac = nextac
  nextk = lastk + Int32(1)
  nextac = _ac(dx, nextk, l, s)

  while nextac > low_cutoff
    push!(lags, lastk)
    push!(acs, lastac)
    lastk = nextk; lastac = nextac
    nextk = lastk + Int32(1)
    nextac = _ac(dx, nextk, l, s)
  end

  # append the first below-low_cutoff measurement
  push!(lags, lastk)
  push!(acs, lastac)

  return (lags, acs)
end

# Linear fit quick implementation (standard least squares)
function linear_fit_coeffs(x::Vector{Float64}, y::Vector{Float64})
  X = hcat(ones(Float64,length(x)), x)
  β = X \ y
  return β[1], β[2]
end

function process_file(sim_data::Dict{String,Any}, plotDir::String)
  # ks
  am_k = get(sim_data, "am_k", nothing)
  m2_k = get(sim_data, "m2_k", nothing)

  k::Int32 = 0

  # compute k = 10 * max(am_k, m2_k); fallback 10000
  if !(am_k isa Number) && !(m2_k isa Number)
    k = 10000
  else
    max_raw = maximum([ am_k isa Number ? am_k : -Inf,
                        m2_k isa Number ? m2_k : -Inf ])
    if max_raw == -Inf
      k = 10000
    else
      kval = floor(Int32,max_raw)
      k = (kval > 0) ? kval : 1
    end
  end

  # read data.csv (assume first row is header, comma sep)
  datafile = joinpath(sim_data["folder"], "data.csv")
  if !isfile(datafile)
    @warn "data.csv not found for descriptor" datafile
    sim_data["tau_exp"] = NaN
    sim_data["tau_int"] = NaN
    return sim_data
  end

  # readdlm returns a matrix; skip header row
  raw = readdlm(datafile, ',', skipstart=1)
  if size(raw,2) < 1
    @warn "data.csv has no columns" datafile
    sim_data["tau_exp"] = NaN
    sim_data["tau_int"] = NaN
    return sim_data
  end

  # extract first column (magnetization)
  am = abs.( Float64.(raw[:,1]) )

  # -------- tau_exp via autocorrelation fit (linear in log-log) ----------
  lags, corr = autocorrelation(am)

  ln_corr = log.(corr)
  a, b = linear_fit_coeffs(Float64.(lags),ln_corr)
  tau_exp = -1.0 / b
  sim_data["tau_exp"] = tau_exp

  # -------- blocking & tau_int -------------------------------------------
  # exclude thermalization: start from 5*tau_exp (if finite)
  start_idx = Int(floor(1 + 5 * sim_data["tau_exp"]))  # 1-based indexing
  start_idx = max(start_idx, 1)

  if start_idx > length(am)
    @warn "start index after end of data" start_idx length(am)
    sim_data["tau_int"] = NaN
    return sim_data
  end

  am_cut = am[start_idx:end]
  n_blocked = length(am_cut) ÷ k
  if n_blocked < 1
    @warn "Too few data points for blocking (k large?)" length(am_cut) k
    sim_data["tau_int"] = NaN
    return sim_data
  end

  blocked_am = zeros(Float64, n_blocked)
  for i in 1:n_blocked
    s = (i-1)*k + 1
    e = i*k
    blocked_am[i] = mean(am_cut[s:e])
  end

  sigma_naive = var(am_cut)
  sigma_real  = var(blocked_am) / n_blocked
  tau_int = 0.5 * ( (n_blocked * k) * sigma_real / sigma_naive - 1.0 )
  sim_data["tau_int"] = tau_int

  # plot
  try
    mkpath(plotDir)

    # autocorrelation plot
    p1 = plot(
      lags, corr,
      xlabel = "lag",
      ylabel = "C(lag)",
      title  = "Autocorrelation",
      marker = :circle, ms=0.3, lw=1.0,
      yscale = :log10,
    )
    llag = [ i/200*minimum(lags)+(200-i)/200*maximum(lags) for i in range(-1,201) ]
    plot!(p1,
      llag,
      exp.(a .+ b .* llag),
    )
    savefig(
      p1,
      joinpath(plotDir, "autocorr_" * basename(sim_data["folder"]) * ".svg")
    )
  catch err
    @warn "plotting in process_file failed" err
  end

  return sim_data
end

#= parse_arguments() =#
function parse_arguments()
  s = ArgParse.ArgParseSettings()
  @add_arg_table! s begin
    "descriptors"
      help = "JSON descriptor files of simulations to analyze"
      nargs = '+'
      required = true
      arg_type = String
    "--outPlotDir"
      help = "Output plot directory"
      required = true
      arg_type = String
    "--outFile"
      help = "Output text file with fit results"
      required = true
      arg_type = String
  end
  return parse_args(s)
end

#= power_fit =#
function power_fit(Ls::Vector{Float64}, taus::Vector{Float64})
  logL = log.(Ls)
  logT = log.(taus)
  a, b = linear_fit_coeffs(logL, logT)
  return exp(a), b
end

#= main =#
function (@main)(args)
  parsed   = parse_arguments()
  plot_dir = parsed["outPlotDir"]
  out_file = parsed["outFile"]
  files    = parsed["descriptors"]

  # create directory (if not exists)
  if !isdir(plot_dir)
    mkpath(plot_dir)
  end

  final_results = Dict{Any,Any}()

  #= Multithread =#
  results = Vector{Dict{String,Any}}()
  my_lock = ReentrantLock()
  nfiles = length(files)

  pbar = ProgressBar(total=nfiles, printing_delay=0.5)

  @threads for idx in 1:nfiles
    fname = files[idx]
    sim_data = try
      JSON.parsefile(fname)
    catch err
      @warn "Failed to parse JSON descriptor" fname err
      continue
    end

    sim_out = process_file(sim_data, plot_dir)

    # push
    lock(my_lock) do
      push!(results, sim_out)
      update(pbar)
    end
  end

  # single thread operations
  colormap = cgrad(:winter)
  known_geoms = ["hex", "sqr", "tri"]
  colors_geo  = Dict{String,Any}()
  for (i,g) in enumerate(known_geoms)
      colors_geo[g] = colormap[i / (length(known_geoms) + 2)]
  end

  fit_data = Dict{String,Any}()

  # group by algorithm
  algorithms = unique([ d["algorithm"] for d in results ])
  for alg in algorithms
    data_alg = filter(d -> d["algorithm"] == alg, results)
    fit_data[alg] = Dict{String,Any}()

    geoms = unique([ d["geometry"] for d in data_alg ])
    for geom in geoms
      data_ag = filter(d -> d["geometry"] == geom, data_alg)

      # arrays L, tau_int, tau_exp
      Ls = [ Float64(d["L"]) for d in data_ag ]
      LL = range(1.05 * minimum(Ls) - 0.05 * maximum(Ls), stop = 1.05 * maximum(Ls) - 0.05 * minimum(Ls), length=200)

      tau_ints = [ Float64(get(d, "tau_int", NaN)) for d in data_ag ]
      tau_exps = [ Float64(get(d, "tau_exp", NaN)) for d in data_ag ]

      good_int = .!isnan.(tau_ints)
      good_exp = .!isnan.(tau_exps)

      # fit tau_int in log-log (linear)
      zprime = NaN
      A_int = NaN
      x_int = Float64.(Ls[good_int])
      y_int = Float64.(tau_ints[good_int])
      try
          A_int, zprime = power_fit(x_int, y_int)
      catch err
          @warn "power fit tau_int failed" err
      end

      # fit tau_exp in log-log (lineare)
      z = NaN
      A_exp = NaN
      x_exp = Float64.(Ls[good_exp])
      y_exp = Float64.(tau_exps[good_exp])
      try
        A_exp, z = power_fit(x_exp, y_exp)
      catch err
        @warn "power fit tau_exp failed" err
      end

      fit_data[alg][geom] = Dict("tau_int" => zprime, "tau_exp" => z)

      # append to output text file
      final_results[alg, geom] = fit_data[alg][geom]

      # plotting tau_exp (log-log plot)
      try
        p = plot(xscale=:log10, yscale=:log10)
        scatter!(p, Ls, tau_exps, label=geom, marker=:circle, ms=4, color=get(colors_geo, geom, :black))
        if isfinite(z)
            plot!(p, LL, A_exp .* (LL .^ z), linestyle=:dash, label="$(geom) fit")
        end
        xlabel!(p, L"$L$")
        ylabel!(p, L"\tau_{exp}")
        title!(p, "$(alg) $(geom)")
        savefig(p, joinpath(plot_dir, @sprintf("tau_exp_%s_%s.svg", alg, geom)))
      catch err
        @warn "plot tau_exp failed" err
      end

      # plotting tau_int (log-log)
      try
        p2 = plot(xscale=:log10, yscale=:log10)
        scatter!(p2, Ls, tau_ints, label=geom, marker=:circle, ms=4, color=get(colors_geo, geom, :black))
        if isfinite(zprime)
          plot!(p2, LL, A_int .* (LL .^ zprime), linestyle=:dash, label="$(geom) fit")
        end
        xlabel!(p2, L"$L$")
        ylabel!(p2, L"\tau_{int}^{|m|}")
        title!(p2, "$(alg) $(geom)")
        savefig(p2, joinpath(plot_dir, @sprintf("tau_int_%s_%s.svg", alg, geom)))
      catch err
        @warn "plot tau_int failed" err
      end
    end
  end

  open(out_file, "w") do io
    JSON.print(io, final_results, 2)
  end

  println("Done. Results written to: ", out_file)
end
