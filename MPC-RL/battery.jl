include("battery_setup.jl")

using Statistics
state_dim = 8
inner_state_dim = Ncp+Ncn+4+Nsei+Ncum
mutable struct FR_battery{S,A,R} 
    current::Vector{S}
    observationspace::ObservationSpace{S}
    actionspace::ActionSpace{A}
    state::Vector{S}

    function FR_battery{T}(state_dim, inner_state_dim) where T <: AbstractFloat
        observationspace = ObservationSpace{T}(state_dim)
        actionspace = ActionSpace([(-one(T), one(T)),(-one(T),one(T))])
        new{T, T, T}(zeros(T,state_dim), observationspace, actionspace, zeros(T,inner_state_dim))
    end
end

FR_battery() = FR_battery{Float64}()

umax = [P_nominal*maxC, P_nominal*maxC]
umin = [0, -P_nominal*maxC]

function denormalizeU(u,ind=0)
    if ind==0
        return (umax .+ umin)/2 .+ (umax .- umin) ./ 2 .* u
    else
        return (umax[ind] .+ umin[ind])/2 .+ (umax[ind] .- umin[ind]) ./ 2 .* u[ind]
    end
end

function step!(env, action::Array{A,1}, FR_price_segment, grid_price_segment, signal_segment, hour::Int64) where {S,A,R}
    
    u0 = env.state 
    ctr = denormalizeU(vec(action))
    FR_band = ctr[1]
    grid_band = ctr[2]
    soc_end, capacity_remain, u0, reward1, reward2, done = step_battery(FR_price_segment, grid_price_segment, signal_segment, FR_band, grid_band, u0, hour)
    
    env.state = u0
    env.current[5:6] .= soc_end, capacity_remain
    return env.current, reward1, reward2, done
end

function reset!(env; start = u_start)
    env.state = start
    env.current = [0,0,0,0,0.5,1,0,0] 
end

function set!(env, FR_price, grid_price, signal, hour)
    env.current[1:4] .= Statistics.mean(signal), Statistics.var(signal), FR_price[1], grid_price[1]
    max_val = P_nominal * maxC  # Reuse from battery_setup.jl (assume maxC is defined there or globally)
    
    # Normalize FR_band_mpc to [-1, 1]: FR is [0, max_val] -> [-1, 1]
    fr_mpc = RLAlgorithms.FR_band_mpc_list[hour]
    env.current[7] = (2 * fr_mpc / max_val) - 1
    
    # Normalize grid_band_mpc to [-1, 1]: grid is [-max_val, max_val] -> [-1, 1]
    grid_mpc = RLAlgorithms.grid_band_mpc_list[hour]
    env.current[8] = grid_mpc / max_val
end
