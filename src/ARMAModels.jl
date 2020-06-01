module ARMAModels
export get_model
export aicbic
export fit!
export predict!
export lag
export lags

using Statistics: var, mean
using Optim
import LineSearches
using LinearAlgebra
using VariableTransforms
import ForwardDiff

struct CurrentSample
    data::Array
    additional_data::Array
    T::Int
    N::Int
end
struct ARMAResults
    yhat::AbstractArray
    LLavg::Number
    AICavg::Number
    BICavg::Number
    parameters::AbstractArray
    transformed_parameters::AbstractArray
    other::Dict
end
mutable struct ARMAModel
    data::CurrentSample
    p::Int
    q::Int
    results::ARMAResults
    storage::Dict
end

function lag(x, l)
    [fill(NaN, l);x[1:end-l]]
end
function lags(x, ls::AbstractArray)
    hcat([lag(x,l) for l in ls]...)
end
function lags(x, ls::Number)
    hcat([lag(x,l) for l in 1:ls]...)
end

function aicbic(llavg::Number,no_params::Number,T::Number; t=:AIC)
    if t==:AIC
#         return 2*no_params-2*T*llavg
        return 2*no_params/T-2*llavg
    else
#         return log(T)no_params-2*T*llavg
        return log(T)*no_params/T-2*llavg
    end
end
aicbic(llavg,params::Array,T; kwargs...)=aicbic(llavg,length(params),T; kwargs...)

function get_model(y,p,q; additional_data=[])
    dtype=typeof(y[1])
    y_lags=lags(y, p)
    #Construct YX
    YX=convert(Array{dtype}, [y ones(length(y)) y_lags])
    if length(additional_data)!=0
        YX=[YX convert(Array{dtype}, additional_data)]
    end
    d=CurrentSample(y,YX[:,2:end],size(YX,1),1)
    #Drop missing
    ix=vec(sum(isnan.(YX);dims=2).==0);
    if q==0
        return ARMAModel(d, p, q, ARMAResults([],-Inf,Inf,Inf,[],[],Dict()),Dict(:Y=>YX[ix,1:1], :X=>YX[ix,2:end], :yhat=>zeros(sum(ix),1), :εhat=>zeros(sum(ix),1), :ix=>ix))
    else
        start_ix=maximum([1+p,1+q,1+sum(maximum(isnan.(y_lags).==1; dims=2))])
        if length(additional_data)!=0
            start_ix=maximum([start_ix,1+sum(maximum(isnan.(additional_data).==1; dims=2))])
        end
        return ARMAModel(d, p, q, ARMAResults([],-Inf,Inf,Inf,[],[],Dict()),Dict(
                :Y=>YX[:,1:1],
                :X=>YX[:,2:end],
                :yhat=>zeros(dtype, sum(d.T),1),
#                 :yhat=>fill(ForwardDiff.Dual, sum(d.T),1),
                :εhat=>zeros(dtype,sum(d.T),1),
#                 :εhat=>fill(ForwardDiff.Dual, sum(d.T),1),
                :ix=>ix,
                :start_ix=>start_ix,
        ))
    end
end
function fit_AR!(model::ARMAModel, parameters; return_results=false)
    Y=view(model.storage[:Y], :, :)
    X=view(model.storage[:X], :, :)

    β=0
    if return_results==false
        β=factorize(Symmetric(X'X))\(X'Y)
    else
        β=model.results.parameters[1:end-1]
    end

    mul!(model.storage[:yhat], X, β)
    model.storage[:εhat]=Y-model.storage[:yhat]

    σ2=var(model.storage[:εhat]; corrected=false)

    SSR=selfdot(model.storage[:εhat])
    T=length(Y)

    llavg=-0.5*log(2π)-0.5*log(σ2)-0.5*(SSR/T)/σ2

    yhat_out=fill(NaN, model.data.T)
    yhat_out[model.storage[:ix], :]=model.storage[:yhat]
    model.results=ARMAResults(
        yhat_out,
        llavg,
        aicbic(llavg,size(X,2)+1,T; t=:AIC),
        aicbic(llavg,size(X,2)+1,T; t=:BIC),
        [β;VariableTransforms.from_pos_to_R(σ2)],
        [β;σ2],
        Dict(
             :μ=>β[1],
             :φ=>β[2:model.p+1],
             :θ=>[],
             :β=>β[model.p+2:end],
             :σ2=>σ2,
        )
    )
end
function fit!(model::ARMAModel; parameters=[], optimizers = [(Optim.Newton, LineSearches.MoreThuente())], autodiff=:forward, sanity_bound=1e6, ND_iterations=0, options=Optim.Options())
    #reset_timer!(tos)
    if model.q==0
        return fit_AR!(model, [])
    else
        if length(parameters)==0
            #Initial values from AR(p)
            modelAR=get_model(model.data.data, model.p, 0; additional_data=model.data.additional_data)
            fit_AR!(modelAR, [])
            parameters=modelAR.results.parameters
            parameters=[parameters[1:model.p+1];zeros(model.q);parameters[model.p+2:end]]
        end
        criterion     = (parameters) -> -ARMA_criterion!(model, parameters; sanity_bound=sanity_bound)
        opt_criterion = Optim.TwiceDifferentiable(criterion, parameters; autodiff=autodiff)

        if ND_iterations>0
            resND=Optim.optimize(
                criterion,
                parameters,
                Optim.NelderMead(),
                Optim.Options(iterations=100)
            )
            parameters = resND.minimizer
        end
        #@timeit tos "Optimizer" begin
        converged=false
        i=1
        res=0
        while i<=length(optimizers) && converged==false
            res=Optim.optimize(
                opt_criterion,
                parameters,
                optimizers[i][1](linesearch = optimizers[i][2]),
                options
            )
            i+=1
            parameters=res.minimizer
            converged=any([getfield(res, f) for f in [:x_converged, :f_converged, :g_converged]])
        end
        #end
        return ARMA_criterion!(model, res.minimizer; return_results=true, res=res)
    end
end
function predict!(sourceModel::ARMAModel,targetModel::ARMAModel)
    targetModel.results=sourceModel.results
    if sourceModel.q==0
        fit_AR!(targetModel, targetModel.results.parameters; return_results=true)
    else
        ARMA_criterion!(targetModel, targetModel.results.parameters; return_results=true)
    end
    targetModel
end
predict!(sourceModel::ARMAModel)=predict!(sourceModel::ARMAModel,sourceModel::ARMAModel)
function selfdot(x::AbstractArray)
    dot(x,x)
end
function ARMA_criterion!(model::ARMAModel, parameters::AbstractArray; return_results=false, res=nothing, sanity_bound=1e6)
#     @timeit to "ARMA_criterion!" begin
    μ=parameters[1]
    φ=parameters[2:1+model.p];
    θ=parameters[1+model.p+1:1+model.p+model.q]; reverse!(θ)
    β=parameters[1+model.p+model.q+1:end-1];
    σ2=parameters[end]; σ2=VariableTransforms.from_R_to_pos(σ2)

    model.storage[:yhat]=zeros(typeof(μ), size(model.storage[:yhat])...)
    model.storage[:εhat]=zeros(typeof(μ), size(model.storage[:εhat])...)
    mul!(model.storage[:yhat], view(model.storage[:X], :, :), [μ;φ;β])
    model.storage[:εhat].+=model.data.data
    model.storage[:εhat][1:model.storage[:start_ix]-1].=0

    @inbounds for t=model.storage[:start_ix]:model.data.T
        model.storage[:εhat][t]-=model.storage[:yhat][t]+=dot(model.storage[:εhat][t-model.q:t-1],θ)
    end

#     @inbounds for t=model.storage[:start_ix]:model.data.T
#         model.storage[:yhat][t]+=dot(model.storage[:εhat][t-model.q:t-1],θ)
#         model.storage[:εhat][t]=model.data.data[t] - model.storage[:yhat][t]
#     end

    SSR_T=model.data.T-model.storage[:start_ix]+1
    SSR=selfdot(model.storage[:εhat])

    llavg=-0.5*log(2π)-0.5*log(σ2)-0.5*(SSR/SSR_T)/σ2
    if return_results==true
        model.storage[:yhat][1:model.storage[:start_ix]-1,:].=NaN
        return model.results=ARMAResults(
            model.storage[:yhat],
            llavg,
            aicbic(llavg,length(parameters),SSR_T; t=:AIC),
            aicbic(llavg,length(parameters),SSR_T; t=:BIC),
            parameters,
            [μ;φ;reverse(θ);β;σ2],
            Dict(
                 :res=>res,
                 :μ=>μ,
                 :φ=>φ,
                 :θ=>reverse(θ),
                 :β=>β,
                 :σ2=>σ2,
            )
        )
    end
#     end
#     display([model.data.data yhat ε][.~ ix,:][1:20,:]')
#     display([SSR SSR_T σ2])
    llavg
end

end # module