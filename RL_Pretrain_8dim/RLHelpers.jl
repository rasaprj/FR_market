module RLHelpers
    import Distributions.sample
    import Base.append!
    using Statistics: mean, std

    include("ReplayMemory.jl")

    function softupdate!(target::T, model::T, τ=1f-2) where T
        for f in fieldnames(T)
            softupdate!(getfield(target, f), getfield(model, f), τ)
        end
    end

    function softupdate!(dst::A, src::A, τ=T(1f-2)) where {T, A<:AbstractArray{T}}
        dst .= τ .* src .+ (one(T) - τ) .* dst
    end

    function soft_update!(target::T, model::T, τ=1f-2) where T
        for i in 1:length(target.layers)
            target.layers[i].W[:] = τ .* target.layers[i].W[:] + (1-τ) .* model.layers[i].W[:]
            target.layers[i].b[:] = τ .* target.layers[i].b[:] + (1-τ) .* model.layers[i].b[:]
        end
    end

    export softupdate!, soft_update!, ReplayMemory
end
