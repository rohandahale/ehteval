"""
Julia Version 1.10.9
"""

using Pkg; Pkg.activate(@__DIR__)
using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)
using ArgParse
using DelimitedFiles
using DataFrames
using CSV
using Comrade
using VIDA
using OptimizationMetaheuristics: OptimizationMetaheuristics, ECA, Options
using VLBISkyModels
using Statistics
using Optimization


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--input"
            help = "Input HDF5 movie file"
            arg_type = String
            required = true
        "--outname"
            help = "output csv file with extractions"
            arg_type = String
            required = true
        "--stride"
             help = "Checkpointing stride, i.e. number of steps."
             arg_type = Int
             default = 8
        "--fevals"
             help = "The number of evaluations allowed when extracting params"
             arg_type = Int
             default = 20000
        "--blur"
             help = "Blur images before extracting fitted model parameters"
             arg_type = Float64
             default = 0.0
        "--restart"
            help = "Tells the sampler to read in the old frames and restart the run."
            action = :store_true
        "--regrid"
            help = "To not regrid the images before extracting"
            action = :store_true
        "--order"
            help = "The order of the ring parameters"
            arg_type = Int
            default = 4
        "--procs"
            help = "Number of worker processes to add"
            arg_type = Int
            default = 8
    end
    return parse_args(s)
end


parsed_args = parse_commandline()
if parsed_args["procs"] > 0
    addprocs(parsed_args["procs"])
end

@everywhere begin
    using ArgParse
    using Distributed
    using DelimitedFiles
    using DataFrames
    using CSV
    using Comrade
    using VIDA
    using OptimizationMetaheuristics: OptimizationMetaheuristics, ECA, Options
    using VLBISkyModels
    using Statistics
    using Optimization
end


@everywhere begin

    function lpmodes(img::IntensityMap{<:StokesParams}, modes::NTuple{<:Any, Int}; rmin=0, rmax=μas2rad(60.0))
        I = stokes(img, :I)
        lp = linearpol.(img)
        g = domainpoints(img)
        r = splat(hypot).(values.(g))
        ang(p) = atan(-p.X, p.Y)
        θ = ang.(g)

        mask = (r .<= rmax).&(r .>= rmin)
        flux = sum(I[mask])
        betas = map(modes) do m
            pbasis = cis.(m.*θ)
            prod = lp.*pbasis
            coef = sum(prod[mask])
            return coef/flux
        end
        return betas
    end

    function cpmodes(img::IntensityMap{<:StokesParams}, modes::NTuple{<:Any, Int}; rmin=0, rmax=μas2rad(60.0))
        I = stokes(img, :I)
        lp = stokes(img, :V)
        g = domainpoints(img)
        r = splat(hypot).(values.(g))
        ang(p) = atan(-p.X, p.Y)
        θ = ang.(g)

        mask = (r .<= rmax).&(r .>= rmin)
        flux = sum(I[mask])
        betas = map(modes) do m
            pbasis = cis.(m.*θ)
            prod = lp.*pbasis
            coef = sum(prod[mask])
            return coef/flux
        end
        return betas
    end

    mnet(img::AbstractArray{<:StokesParams}) = abs(sum(linearpol, img))/sum(stokes(img, :I))
    vnet(img::AbstractArray{<:StokesParams}) = sum(stokes(img, :V))/sum(stokes(img, :I))
    mavg(img::AbstractArray{<:StokesParams}) = sum(abs.(linearpol.(img)))/sum(stokes(img, :I))
    vavg(img::AbstractArray{<:StokesParams}) = sum(abs.(stokes(img, :V)))/sum(stokes(img, :I))

    netevpa(img::AbstractArray{<:StokesParams}) = evpa(mean(img))

    function polnxcorr(x::IntensityMap{<:StokesParams{T}}, y::IntensityMap{<:StokesParams}) where {T}
        sp = zero(Complex{T})
        sv = zero(T)
        nx = zero(T)
        ny = zero(T)
        nvx= zero(T)
        nvy= zero(T)
        for I in eachindex(x, y)
            xx = x[I]
            yy = y[I]
            lpx = linearpol(xx)
            lpy = linearpol(yy)
            sp += lpx*conj(lpy)
            nx += abs2(lpx)
            ny += abs2(lpy)
            sv += xx.V*yy.V
            nvx += abs2(xx.V)
            nvy += abs2(yy.V)
        end
        return (LP = real(sp)*inv(sqrt(nx*ny)),  V =sv*inv(sqrt(nvx*nvy)))
    end

    function VIDA.nxcorr(x::IntensityMap{<:StokesParams}, y::IntensityMap{<:StokesParams}) 
        nI = nxcorr(stokes(x, :I), stokes(y, :I))
        nP = polnxcorr(x, y)
        return merge((I=nI,), nP)
    end

    function center_template(img, template::Type;
        grid=axisdims(img),
        div=NxCorr,
        maxiters=10_000)
        if !isnothing(grid)
            rimg = regrid(img, grid)
        else
            rimg = img
        end
        xopt, θopt = _center_template(rimg, template, div, maxiters)
        return shifted(img, -xopt.x0, -xopt.y0), xopt, θopt
    end

    function center_template(img::IntensityMap{<:StokesParams}, template::Type;
        grid=axisdims(img),
        div=NxCorr,
        maxiters=10_000)
        _, xopt, θopt = center_template(stokes(img, :I), template; grid, div, maxiters)
        return shifted(img, -xopt.x0, -xopt.y0), xopt, θopt
    end

    function _optimize(div, func, lower, upper, p0, maxiters=8_000)
        prob = VIDAProblem(div, func, lower, upper)
        xopt, θopt, dmin = VIDA.vida(prob, ECA(; options=Options(f_calls_limit=maxiters, f_tol=1e-5)); init_params=p0)
        return merge(xopt, (; divmin=dmin)), θopt
    end

    # Disk model support
    # (Simplified for MRing use case, but defining _center_template for MRing)

    function _center_template(img::IntensityMap{<:Real}, ::Type{<:MRing{N}}, div, maxiters) where {N}
        bh = div(max.(img, 0.0))
        x0, y0 = centroid(img)

        # Using VLBISkyModels explicitly as in library
        temp(x) =
            modify(RingTemplate(RadialGaussian(x.σ / x.r0), AzimuthalCosine(x.s, x.ξ .- x.ξτ)),
                Stretch(x.r0, x.r0 * (1 + x.τ)), Rotate(x.ξτ), Shift(x.x0, x.y0)) +
            x.f0 * VLBISkyModels.Constant(fieldofview(img).X)
            
        lower = (r0=μas2rad(10.0), σ=μas2rad(0.5),
            s=ntuple(_ -> 0.001, N),
            ξ=ntuple(_ -> 0.0, N),
            τ=0.0,
            ξτ=0.0,
            x0=-μas2rad(20.0), y0=-μas2rad(20.0),
            f0=1e-6
        )
        upper = (r0=μas2rad(60.0), σ=μas2rad(25.0),
            s=ntuple(_ -> 0.999, N),
            ξ=ntuple(_ -> 2π, N),
            τ=1.0,
            ξτ=1π,
            x0=μas2rad(20.0), y0=μas2rad(20.0),
            f0=10.0
        )
        p0 = (r0=μas2rad(16.0), σ=μas2rad(4.0),
            s=ntuple(_ -> 0.2, N),
            ξ=ntuple(_ -> 1π, N),
            τ=0.01,
            ξτ=0.5π,
            x0=x0, y0=y0,
            f0=0.1)
        return _optimize(bh, temp, lower, upper, p0, maxiters)
    end

    function center_ring(img::IntensityMap; order=1, g=axisdims(img), maxiters=10_000)
        return center_template(img, MRing{order}, grid=g, maxiters=maxiters)
    end

    function summary_ringparams(img::IntensityMap{<:StokesParams};
                                lpmode=(1, 2,), cpmode=(1,),
                                order=3, maxiters=20_000, 
                                divergence=NxCorr,
                                grid = nothing,
                                cfluxdiam = μas2rad(80.0))
        xopt = summary_ringparams(stokes(img, :I); order, grid, maxiters, cfluxdiam, divergence)
        rx = cfluxdiam/2
        
        simg = shifted(img, -xopt.x0, -xopt.y0)

        simg = simg[X=-rx..rx, Y=-rx..rx]
        
        cflux = flux(simg)

        m_net = mnet(simg)
        v_net = vnet(simg)

        m_avg = mavg(simg)
        v_avg = vavg(simg)

        βlp = lpmodes(simg, lpmode; rmax=rx)
        βcp = cpmodes(simg, cpmode; rmax=rx)

        n_relp = Tuple((Symbol("re_betalp", "_$n") for n in lpmode))
        real_betalp = NamedTuple{n_relp}(map(real, βlp))
        n_imlp = Tuple((Symbol("im_betalp", "_$n") for n in lpmode))
        imag_betalp = NamedTuple{n_imlp}(map(imag, βlp))

        n_recp = Tuple((Symbol("re_betacp", "_$n") for n in cpmode))
        real_betacp = NamedTuple{n_recp}(map(real, βcp))
        n_imcp = Tuple((Symbol("im_betacp", "_$n") for n in cpmode))
        imag_betacp = NamedTuple{n_imcp}(map(imag, βcp))

        nevpa = netevpa(img)

        return merge(xopt, (;Qtot = cflux.Q, Utot = cflux.U, Vtot=cflux.V), 
                            (;evpa=nevpa), (;m_net, m_avg), 
                            real_betalp, imag_betalp, (;v_net, v_avg), 
                            real_betacp, imag_betacp)
    end

    function summary_ringparams(img::IntensityMap{<:Real};
                                lpmode=(2,), cpmode=(1,),
                                order=3, maxiters=20_000,
                                divergence=NxCorr, 
                                grid = nothing,
                                cfluxdiam=μas2rad(80.0))
        _, xopt, _ = center_template(img, MRing{order}; maxiters, grid=grid, div=divergence)
        rx = cfluxdiam/2
        
        simg = shifted(img, -xopt.x0, -xopt.y0)
        simg = simg[X=-rx..rx, Y=-rx..rx]
        
        cflux = flux(simg)
        return merge(_flatten_tuple(xopt), (;Itot=cflux))
    end

    function _flatten_tuple(nt::NamedTuple)
        names = keys(nt)
        vals = values(nt)
        mapreduce(merge, zip(names, vals)) do (name, value)
            return _flatten_tuple(name, value)
        end
    end

    function _flatten_tuple(name::Symbol, a)
        return NamedTuple{(name,)}((a,))
    end

    function _flatten_tuple(name::Symbol, t::NTuple{N}) where {N}
        names = Tuple((Symbol("$(name)_$i") for i in 1:N))
        return NamedTuple{names}(t)
    end

end

function main()
    infile = parsed_args["input"]
    outname = parsed_args["outname"]
    stride = parsed_args["stride"]
    blur = parsed_args["blur"]
    restart = parsed_args["restart"]
    regrid = parsed_args["regrid"]
    order = parsed_args["order"]
    fevals = parsed_args["fevals"]

    @info "Input file: $(infile)"
    @info "Outputting results to $(outname)"
    @info "Using a $(order) order ring model"

    g = imagepixels(μas2rad(200.0), μas2rad(200.0), 64, 64)

    # Flip this because by default we want to regrid
    #--regrid flag means "Do NOT regrid".
    regrid = !regrid

    @info "Regridding image : $(regrid)"
    @info "Blurring kernel: $(blur) μas"
    
    # Load HDF5 movie
    movie = load_hdf5(infile, ; polarization = true)
    times = get_times(movie)
    ntimes = length(times)
    @info "Loaded movie with $(ntimes) frames"

    @info "Starting to extract summary statistics"
    if restart && isfile(outname)
        df = CSV.read(outname, DataFrame)
        startindex = nrow(df)+1
    else
        df = DataFrame()
        startindex = 1
    end
    
    if startindex > ntimes
         @info "All frames processed."
         return
    end

    indexpart = Iterators.partition(startindex:ntimes, stride)
    for ii in indexpart
        @info "Extracting from $(ii[begin]) to $(ii[end])"
        
        # Parallel map
        res = pmap(ii) do i
             t_val = times[i]
             img = get_image(movie, t_val)
             
             # Center image
             cimg = center_image(img)
                
             if regrid
                 rimg = Comrade.regrid(cimg, g)
             else
                 rimg = cimg
             end

             if blur > 0.0
                 rimg = Comrade.smooth(rimg, μas2rad(blur)/(2*sqrt(2*log(2))))
             end
             
             stats = summary_ringparams(rimg; 
                                        lpmode=(1,2,), 
                                        cpmode=(1,),
                                        order=order, 
                                        maxiters=fevals,
                                        cfluxdiam = μas2rad(80.0))
             # Free intermediate images on worker to prevent memory buildup
             cimg = nothing
             rimg = nothing
             img = nothing
             GC.gc()
             return stats
        end
        
        dftmp = DataFrame(res)
        dftmp.time = times[ii] # Add time column
        df = vcat(df, dftmp)
        CSV.write(outname, df)
        
        # Trigger garbage collection on main process and all workers
        GC.gc()
        @sync for w in workers()
            @async remotecall_wait(GC.gc, w)
        end
    end

    # Properly remove worker processes to free memory
    @info "Cleaning up worker processes..."
    rmprocs(workers()...)
    GC.gc()
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
