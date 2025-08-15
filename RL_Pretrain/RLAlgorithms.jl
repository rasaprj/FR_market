module RLAlgorithms
    global FR_band_mpc_list
    global grid_band_mpc_list
    include("RLEnvironments.jl")
    include("RLHelpers.jl")
    using Reexport
    @reexport using .RLHelpers
    @reexport using .RLEnvironments
    using StatsBase: sample, Weights
    include("battery.jl")


    export episode!, episodes!, test

    function episodes!(f, memory, env, steps, maxt, FR_price_total, grid_price_total, signal_segment_total)
        Nt_FR_hour =  round(Int, 3600/2)
        j = 0
        done = false
        total_r1 = 0.0
        total_r2 = 0.0 
        while j < steps 
            jj, done, r1_sum, r2_sum = episode!(f, memory, env, min(maxt, steps-j), 
                FR_price_total[j+1:j+maxt], grid_price_total[j+1:j+maxt], 
                    signal_segment_total[j*Nt_FR_hour+1:(j+maxt)*Nt_FR_hour+1], j+1)
            j += jj
            total_r1 += r1_sum
            total_r2 += r2_sum
            done && break
        end
        return done, total_r1, total_r2 
    end

    function episode!(f, memory, env, maxt, FR_price, grid_price, signal, start_hour)
        done = false
        r1_total = 0.0 
        r2_total = 0.0 
        Nt_FR_hour =  round(Int, 3600/2)
        t = 0  
        while t < maxt 
            FR_price_segment = FR_price[t+1]
            grid_price_segment = grid_price[t+1]
            signal_segment = signal[t*Nt_FR_hour+1:(t+1)*Nt_FR_hour+1]
            set!(env, FR_price_segment, grid_price_segment, signal_segment)
            s = copy(env.current)
            a = f(s)

            _, r1, r2, done = step!(env, a, FR_price_segment, grid_price_segment, signal_segment, start_hour + t)
            append!(memory, s, a, r2, done)
            r1_total += r1 
            r2_total += r2 
            t += 1
            done && break
        end
        return t, done, r1_total, r2_total 
    end

    function test(f, env, FR_price_segment, grid_price_segment, signal_segment, maxt=28*24) where {S, R, A}
        reward1 = 0.0
        reward2 = 0.0
        state_copy = copy(env.state)
        current_copy = copy(env.current)
        s = reset!(env)
        Nt_FR_hour = Int(3600/2)
        for i in 1:maxt
            set!(env, FR_price_segment[i], grid_price_segment[i], signal_segment[(i-1)*Nt_FR_hour+1:(i)*Nt_FR_hour+1])
            s = env.current
            _, r1, r2, done = step!(env, f(s), FR_price_segment[i], grid_price_segment[i], signal_segment[(i-1)*Nt_FR_hour+1:(i)*Nt_FR_hour+1], i)  ## step function needs additional information
            reward1 += r1
            reward2 += r2
            done && break
        end
        env.state = state_copy
        env.current = current_copy
        return reward1, reward2
    end

end
