module RLEnvironments
    using Distributions: Uniform
    import Distributions.sample

    abstract type Environment{S,A,R<:Real} end
    abstract type DiscEnv{S,A,R} <: Environment{S,A,R} end
    abstract type ContEnv{S,A,R} <: Environment{S,A,R} end
    abstract type DiffEnv{S,A,R} <: Environment{S,A,R} end
    function step! end
    function reset! end
    function set! end
    struct ActionSpace{A}
        actions::Vector{Tuple{A,A}}
    end


    Base.length(a::ActionSpace) = length(a.actions)
    Base.eltype(::Type{ActionSpace{A}}) where A = A

    function sample(a::ActionSpace{T}) where T <: AbstractFloat
        return map(x -> T(rand(Uniform(x[1], x[2]))), a.actions)
    end

    struct ObservationSpace{S}
        n::Int64
    end

    Base.length(o::ObservationSpace) = o.n
    Base.eltype(::Type{ObservationSpace{S}}) where S = S

    include("battery.jl")
    export Environment, DiscEnv, ContEnv, DiffEnv, step!, reset!, ActionSpace, ObservationSpace, set!
    export FR_battery
end
