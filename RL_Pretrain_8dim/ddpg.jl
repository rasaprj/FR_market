cd("C:\\Users\\ASUS\\Desktop\\FR_market")

using Flux
using Flux: Optimise.update!
using Statistics
using Statistics: mean
using DifferentialEquations
using Sundials
using DelimitedFiles
using Interpolations
using Distributions
using JuMP
using Ipopt
using JLD
using BSON

include("RLAlgorithms.jl") 
using .RLAlgorithms


include("DataLoading.jl")
include("battery.jl") 
include("battery_setup.jl")

Nt_FR_hour = round(Int, 3600/2)

old_state_dim = 6
action_dim = 2

# Create placeholder old models with old dimensions (needed for BSON loading)
old_actor = Chain(
    Dense(old_state_dim => 30, relu),
    Dense(30 => 15, relu),
    Dense(15 => action_dim, tanh)
)  |> gpu  # Uncomment if using GPU

old_critic = Chain(
    Dense(old_state_dim + action_dim => 30, relu),
    Dense(30 => 15, relu),
    Dense(15 => 1)
)  |> gpu

BSON.@load "RL_Pretrain_8dim/actor_model_rl_DA.bson" actor
BSON.@load "RL_Pretrain_8dim/critic_model_rl_DA.bson" critic

old_actor = actor
old_critic = critic

# Now create new models with state_dim=8
state_dim = 8  # Your new dim
actor = Chain(
    Dense(state_dim => 30, relu),
    Dense(30 => 15, relu),
    Dense(15 => action_dim, tanh)
)  |> gpu

critic = Chain(
    Dense(state_dim + action_dim => 30, relu),
    Dense(30 => 15, relu),
    Dense(15 => 1)
)  |> gpu

# Transfer weights: Actor (input changes from 6 to 8)
# Layer 1: Copy first 6 input columns, init extra 2 with glorot_uniform (Flux default)
actor[1].weight[:, 1:old_state_dim] .= old_actor[1].weight
actor[1].weight[:, (old_state_dim+1):state_dim] .= Flux.glorot_uniform(30, state_dim - old_state_dim)  # Random init for new dims
actor[1].bias .= old_actor[1].bias  # Bias unchanged

# Layers 2 and 3: Direct copy
actor[2].weight .= old_actor[2].weight
actor[2].bias .= old_actor[2].bias
actor[3].weight .= old_actor[3].weight
actor[3].bias .= old_actor[3].bias

# Transfer weights: Critic (input changes from 6+2=8 to 8+2=10)
# If critic preload is available; otherwise, skip this block and use fresh critic
critic[1].weight[:, 1:old_state_dim] .= old_critic[1].weight[:, 1:old_state_dim]  # Old states 1-6
critic[1].weight[:, (state_dim+1):(state_dim + action_dim)] .= old_critic[1].weight[:, (old_state_dim+1):(old_state_dim + action_dim)]  # Old actions 7-8 to new 9-10
critic[1].weight[:, (old_state_dim+1):state_dim] .= Flux.glorot_uniform(30, state_dim - old_state_dim)  # Random init for new state dims 7-8
critic[1].bias .= old_critic[1].bias  # Bias unchanged

# Layers 2 and 3: Direct copy
critic[2].weight .= old_critic[2].weight
critic[2].bias .= old_critic[2].bias
critic[3].weight .= old_critic[3].weight
critic[3].bias .= old_critic[3].bias

println("New NN.")

inner_state_dim = Ncp+Ncn+4+Nsei+Ncum

test_horizon = 7*24 # hours
FR_price_test = FR_price_orig[1:test_horizon]
grid_price_test = grid_price_orig[1:test_horizon]
signal_test = signal[1:test_horizon*Nt_FR_hour+1]
epoch_week = 1  # one week corresponds to 168 hours

function act(actor, s, noisescale)
    a = actor(s)
    return max.(-1, min.(1, a .+ noisescale .* randn(Float64,size(a))))
end


sim_data = readdlm("Data/simulation_data.csv", ',', Float64)  

# Extract setpoints (column 1: FR band, column 2: grid buy/sell)
RLAlgorithms.FR_band_mpc_list = sim_data[:, 1]  
RLAlgorithms.grid_band_mpc_list = sim_data[:, 2] 

println("Loaded FR_band_mpc_list length: ", length(RLAlgorithms.FR_band_mpc_list))
println("Loaded grid_band_mpc_list length: ", length(RLAlgorithms.grid_band_mpc_list))
@assert length(RLAlgorithms.FR_band_mpc_list) >= 168 "CSV setpoint list is too short; ensure simulation_data.csv has at least 168 rows"

function run!(actor, aopt, critic, copt, env;
        epochs=75, steps=168*epoch_week, maxt=168, batchsize=160, noisescale=5f-2, γ=0.9, τ=0.01, anchor_weight=0.5f0) ## steps  = 168*4
    rewards1 = zeros(Float64, epochs)
    rewards2 = zeros(Float64, epochs)
    cap_epochs = zeros(Float64, epochs)

    actortar = deepcopy(actor)
    critictar = deepcopy(critic)
    anchor_critic = deepcopy(critic)
    memory = RLHelpers.ReplayMemory{Float64, Float64, Float64}(length(env.observationspace), length(env.actionspace), 10*steps, batchsize)

    actor_opt_state = Flux.setup(aopt, actor)
    critic_opt_state = Flux.setup(copt, critic)

    reset!(env)

    for i in 1:epochs

        FR_price_sample = rand(d_FR_price, 7*epoch_week*24)  
        FR_price_sample = min.(max.(FR_price_sample, min_FR_price), max_FR_price)
        grid_price_sample = (1/50)*vcat( sum(rand(d_grid_price, epoch_week) for i in 1:50)...) 
        signal_start = rand(signal_index) 
        signal_end = signal_start+Nt_FR_hour*steps+1
        signal_segment = signal[signal_start:signal_end]

        done, total_r1, total_r2 = RLAlgorithms.episodes!(memory, env, steps, maxt, FR_price_sample, grid_price_sample, signal_segment) do s 
            act(actor, s, noisescale)
        end

        done && break

        rewards1[i] = total_r1
        rewards2[i] = total_r2
        cap_epochs[i] = env.current[6]
        println("Epoch $i - Reward1: $total_r1, Reward2: $total_r2, CapacityRemain: $(cap_epochs[i])") 

        for _ in 1:steps*4  
            s1, a1, r, s2, done = sample(memory)

            atar2 = actortar(s2)
            Qtars2 = vec(critictar(vcat(s2, atar2)))
            Qtar = r .+ (1 .- done) .* γ .* Qtars2

            cgs = Flux.gradient(critic) do crit
                Q = vec(crit(vcat(s1, a1)))
                return Flux.mae(Q, Qtar)
            end[1]
            update!(critic_opt_state, critic, cgs)

            ags = Flux.gradient(actor) do act
                as = act(s1)
                main_q = critic(vcat(s1, as))
                anchor_q = anchor_critic(vcat(s1, as))
                return -mean(main_q .* (abs.(anchor_q) .^ anchor_weight))
            end[1]
            update!(actor_opt_state, actor, ags)

            softupdate!(actortar, actor, τ)
            softupdate!(critictar, critic, τ)
            #softupdate!(anchor_critic, critic, 0.1 * τ)
        end
        println("finished epoch: ", i)
        if i%5 == 0
    
             save("reward.jld", "epoch", i)
             BSON.@save "actor_model_rl_sp.bson" actor
             BSON.@save "critic_model_rl_sp.bson" critic
        end  
    end

    return rewards1, rewards2, actor, critic, cap_epochs
end

env = FR_battery{Float64}(state_dim, inner_state_dim)

aDim = length(env.actionspace)
oDim = length(env.observationspace)

println("starting...")



start_time = time()
r1, r2, actor, critic = run!(actor, ADAM(5e-5), critic, ADAM(1e-4), env; anchor_weight=0.5f0)

println("training time is :", time() - start_time)

save("reward.jld", "rewards1", r1, "rewards2", r2, "training_time", time()-start_time)

BSON.@save "actor_model_rl_sp.bson" actor
BSON.@save "critic_model_rl_sp.bson" critic

using DelimitedFiles  
using CSV  
using Tables
using Plots
function simulate_and_plot(actor, env, FR_price_segment, grid_price_segment, signal_segment, maxt=365*24)
    time_list = Int[]
    soc_list = Float64[]
    capacity_list = Float64[]
    sei_list = Float64[]
    fr_band_list = Float64[]
    grid_band_list = Float64[]
    reward1_list = Float64[]
    reward2_list = Float64[]
    
    # Initialize MPC reference lists
    fr_mpc_list = Float64[]
    grid_mpc_list = Float64[]

    reset!(env)
    u0 = env.state  # Assuming this is the inner state

    for i in 1:maxt
        seg_start = (i-1) * Nt_FR_hour + 1
        seg_end = i * Nt_FR_hour + 1  # Include +1 to make length 1801
        
        # Fix: Ensure we don't go beyond signal_segment bounds
        if seg_end > length(signal_segment)
            println("Warning: Reached end of signal data at hour $i")
            break
        end
        
        current_signal_segment = signal_segment[seg_start:seg_end]

        set!(env, FR_price_segment[i], grid_price_segment[i], current_signal_segment, i)

        a = actor(env.current)
        ctr = denormalizeU(vec(a))
        FR_band = ctr[1]
        grid_band = ctr[2]

        soc_end, capacity_remain, u0_new, r1, r2, done = step_battery(FR_price_segment[i], grid_price_segment[i], current_signal_segment, FR_band, grid_band, u0, i)
        u0 = u0_new
        env.state = u0
        env.current[5:6] .= soc_end, capacity_remain

        push!(time_list, i)
        push!(soc_list, soc_end)
        push!(capacity_list, capacity_remain)
        push!(sei_list, u0[Ncp+Ncn+7])  # SEI thickness index from inner state
        push!(fr_band_list, FR_band)
        push!(grid_band_list, grid_band)
        push!(reward1_list, r1)
        push!(reward2_list, r2)
        
        # Store MPC reference values for comparison
        if i <= length(RLAlgorithms.FR_band_mpc_list)
            push!(fr_mpc_list, RLAlgorithms.FR_band_mpc_list[i])
            push!(grid_mpc_list, RLAlgorithms.grid_band_mpc_list[i])
        else
            push!(fr_mpc_list, 0.0)  # Default value if MPC list is shorter
            push!(grid_mpc_list, 0.0)
        end

        done && break
    end

    # Save data to CSV 
    headers = ["time", "soc", "capacity_remain", "fr_band", "grid_band", "sei_thickness", "fr_mpc", "grid_mpc", "operational_profit", "total_reward"]
    data_matrix = hcat(time_list, soc_list, capacity_list, fr_band_list, grid_band_list, sei_list, fr_mpc_list, grid_mpc_list, reward1_list, reward2_list)
    
    open("simulation_data_rl.csv", "w") do io
        CSV.write(io, Tables.table(data_matrix, header=headers), delim=',')
    end

    # Plotting
    p1 = plot(1:length(soc_list), soc_list, label="SoC", xlabel="Time (hours)", ylabel="SoC")
    savefig(p1, "soc_plot.png")

    p2 = plot(1:length(capacity_list), capacity_list, label="Capacity Remain", xlabel="Time (hours)", ylabel="Capacity Remain")
    savefig(p2, "capacity_plot.png")

    p3 = plot(1:length(sei_list), sei_list, label="SEI Thickness", xlabel="Time (hours)", ylabel="SEI Thickness")
    savefig(p3, "sei_plot.png")

    p4 = plot(1:length(fr_band_list), fr_band_list, label="FR Band Decisions", xlabel="Time (hours)", ylabel="FR Band")
    savefig(p4, "fr_band_plot.png")

    p5 = plot(1:length(grid_band_list), grid_band_list, label="Grid Buy/Sell Decisions", xlabel="Time (hours)", ylabel="Grid Band (positive: buy, negative: sell)")
    savefig(p5, "grid_band_plot.png")

    p6 = plot(time_list, reward1_list, label="Operational Profit per Hour", xlabel="Time (hours)", ylabel="Profit")
    savefig(p6, "operational_profit_per_hour.png")

    p7 = plot(time_list, reward2_list, label="Total Reward per Hour", xlabel="Time (hours)", ylabel="Reward")
    savefig(p7, "total_reward_per_hour.png")

    println("Plots saved as PNG files.")
    println("Simulation data saved to simulation_data_rl.csv with headers")
    
    return time_list, soc_list, capacity_list, fr_band_list, grid_band_list
end

# After training, load final policy and simulate/plot using test data
BSON.@load "actor_model_rl_sp.bson" actor
println("starting the simulation using the best saved policy:")
sim_horizon = 365 * 24
FR_price_sim = FR_price_orig[1:sim_horizon]
grid_price_sim = grid_price_orig[1:sim_horizon]
signal_sim = signal[1:sim_horizon * Nt_FR_hour + 1]
simulate_and_plot(actor, env, FR_price_sim, grid_price_sim, signal_sim, sim_horizon)
