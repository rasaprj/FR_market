# Hierarchical control strategy: MPC (upper level, degradation-focused) + Pretrained RL (lower level, revenue-focused with setpoint tracking)

using BSON
using CSV
using DelimitedFiles
using Distributions
using Flux
using Interpolations
using Ipopt
using JLD
using JuMP
using LinearAlgebra
using Plots
using Statistics
using Sundials
using Tables

# Include necessary files (reusing existing code)
include("DataLoading.jl")  # Loads FR_price_orig, grid_price_orig, signal, etc.


include("RLEnvironments.jl")
using .RLEnvironments
include("battery.jl")  # FR_battery env, step!, reset!, set!, denormalizeU
include("RLAlgorithms.jl")  # RLEnvironments, episode!, test, FR_band_mpc_list, grid_band_mpc_list
using .RLAlgorithms
include("MPC.jl")  


# Load pretrained RL actor
BSON.@load "actor_model_rl_sp.bson" actor

# Simulation parameters
max_horizon = 365 * 24  # Full year in hours
Nt_FR_hour = round(Int, 3600 / dt_FR)  # Time steps per hour (dt_FR=2s)
FR_price_sim = FR_price_orig[1:max_horizon]
grid_price_sim = grid_price_orig[1:max_horizon]
signal_sim = signal[1:max_horizon * Nt_FR_hour + 1]

# Initialize environment and state
state_dim = 8
inner_state_dim = Ncp + Ncn + 4 + Nsei + Ncum
env = FR_battery{Float64}(state_dim, inner_state_dim)
RLAlgorithms.reset!(env) 
global u0 = env.state

# Logging lists
time_list = Int[]
soc_list = Float64[]
capacity_list = Float64[]
sei_list = Float64[]
fr_band_rl_list = Float64[]
grid_band_rl_list = Float64[]
fr_band_mpc_list_logged = Float64[]
grid_band_mpc_list_logged = Float64[]
profit_list = Float64[]  # Operational profit (reward1)
reward_list = Float64[]  # Total reward (reward2)

# Ensure mpc_lists are initialized (global in RLAlgorithms)
RLAlgorithms.FR_band_mpc_list = zeros(Float64, max_horizon)
RLAlgorithms.grid_band_mpc_list = zeros(Float64, max_horizon)

# Main simulation loop: Run until EOL or max_horizon
global hour = 1
done = false
while hour <= max_horizon && !done
    # Upper level: Run MPC to get setpoints (1-hour horizon, soc_min=0.1, soc_max=0.9)
    global hour, u0
    t_start = (hour - 1) * 3600  # Convert hour to seconds
    FR_band_mpc, grid_band_mpc = OptimalControl(u0, t_start, 0.1, 0.9)  # From FP_MPC.jl
    
    # Set setpoints for RL tracking
    RLAlgorithms.FR_band_mpc_list[hour] = FR_band_mpc
    RLAlgorithms.grid_band_mpc_list[hour] = grid_band_mpc

    # Prepare segment for this hour
    signal_start = (hour - 1) * Nt_FR_hour + 1
    signal_end = hour * Nt_FR_hour + 1
    signal_segment = signal_sim[signal_start:signal_end]

    # Lower level: RL decides action (fr_band_rl, grid_band_rl) based on current state and setpoints
    RLAlgorithms.set!(env, FR_price_sim[hour:hour], grid_price_sim[hour:hour], signal_segment, hour)
    s = env.current
    action = Float64.(actor(s))  # Pretrained policy
    fr_band_rl = denormalizeU(vec(action), 1)
    grid_band_rl = denormalizeU(vec(action), 2)

    # Simulate battery step with RL action (includes degradation via ODE)
    current, r1, r2, done = RLAlgorithms.step!(env, action, FR_price_sim[hour:hour], grid_price_sim[hour:hour], signal_segment, hour)
    u0 = env.state  # Update state for next MPC call

    # Log metrics
    push!(time_list, hour)
    push!(soc_list, current[5])
    push!(capacity_list, current[6])
    push!(sei_list, u0[Ncp + Ncn + 7])
    push!(fr_band_rl_list, fr_band_rl)
    push!(grid_band_rl_list, grid_band_rl)
    push!(fr_band_mpc_list_logged, FR_band_mpc)
    push!(grid_band_mpc_list_logged, grid_band_mpc)
    push!(profit_list, r1)
    push!(reward_list, r2)

    # Check EOL (done from step_battery if capacity_remain <= 0.8)
    if done
        println("Battery reached EOL at hour: $hour")
        break
    end

    hour += 1
end

# Save data to CSV
headers = ["time", "soc", "capacity_remain", "sei_thickness", "fr_band_rl", "grid_band_rl", "fr_band_mpc", "grid_band_mpc", "operational_profit", "total_reward"]
data_matrix = hcat(time_list, soc_list, capacity_list, sei_list, fr_band_rl_list, grid_band_rl_list, fr_band_mpc_list_logged, grid_band_mpc_list_logged, profit_list, reward_list)
open("hierarchical_simulation_data.csv", "w") do io
    CSV.write(io, Tables.table(data_matrix, header=headers), delim=',')
end
println("Simulation data saved to hierarchical_simulation_data.csv")

# Optional: Generate plots
p_soc = Plots.plot(time_list, soc_list, label="SOC", xlabel="Time (hours)", ylabel="SOC")
Plots.savefig(p_soc, "hierarchical_soc_plot.png")

p_capacity = Plots.plot(time_list, capacity_list, label="Capacity Remain", xlabel="Time (hours)", ylabel="Capacity Remain")
Plots.savefig(p_capacity, "hierarchical_capacity_plot.png")

p_sei = Plots.plot(time_list, sei_list, label="SEI Thickness", xlabel="Time (hours)", ylabel="SEI Thickness")
Plots.savefig(p_sei, "hierarchical_sei_plot.png")

p_fr_band = Plots.plot(time_list, fr_band_rl_list, label="FR Band (RL)", xlabel="Time (hours)", ylabel="FR Band")
plot!(p_fr_band, time_list, fr_band_mpc_list_logged, label="FR Band (MPC)", linestyle=:dash)
Plots.savefig(p_fr_band, "hierarchical_fr_band_plot.png")

p_grid_band = Plots.plot(time_list, grid_band_rl_list, label="Grid Band (RL)", xlabel="Time (hours)", ylabel="Grid Band")
plot!(p_grid_band, time_list, grid_band_mpc_list_logged, label="Grid Band (MPC)", linestyle=:dash)
Plots.savefig(p_grid_band, "hierarchical_grid_band_plot.png")

p_profit = Plots.plot(time_list, profit_list, label="Operational Profit", xlabel="Time (hours)", ylabel="Profit")
Plots.savefig(p_profit, "hierarchical_profit_plot.png")

p_reward = Plots.plot(time_list, reward_list, label="Total Reward", xlabel="Time (hours)", ylabel="Reward")
Plots.savefig(p_reward, "hierarchical_reward_plot.png")

println("Plots saved as PNG files.")
println("Hierarchical simulation completed.")