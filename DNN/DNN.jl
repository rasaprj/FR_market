using JLD
using Flux
using BSON: @save

data_file = "TrainData.jld"
loaded_data = load(data_file)

input_train = loaded_data["X"]
output_train = loaded_data["Y"]
stage_costs = loaded_data["COST"]

actor = Chain(Dense(6, 30, relu),
                   Dense(30, 15, relu),
                   Dense(15, 2, tanh))

mae_loss(x, y) = Flux.mae(actor(x), y) 

dataset_size = size(input_train)[2]
training_data = [(input_train[:, i], output_train[:, i]) for i = 1:dataset_size]
optimizer = ADAM()
Flux.@epochs 5000 Flux.train!(mae_loss, Flux.params(actor), training_data, optimizer)
println("Policy network training completed after 5000 epochs") 

@save "actor_model.bson" actor

# Train value network

function sample_data(states, actions, costs, batch_size=24000)
    indices = 1:size(states)[2] - 1
    s_current, a_current, rewards = states[:, indices], actions[:, indices], costs[indices]
    s_next, a_next = states[:, indices .+ 1], actions[:, indices .+ 1]
    return s_current, a_current, rewards, s_next, a_next
end

function soft_update!(target_net::T, source_net::T, tau=1f-2) where T
    for field in fieldnames(T)
        soft_update!(getfield(target_net, field), getfield(source_net, field), tau)
    end
end

function soft_update!(dest::A, src::A, tau=T(1f-2)) where {T, A <: AbstractArray{T}}
    dest .= tau .* src .+ (one(T) - tau) .* dest
end

obs_dim, act_dim = 6, 2
critic = Chain(
    Dense(obs_dim + act_dim, 30, relu),
    Dense(30, 15, relu),
    Dense(15, 1)
)

target_value_net = deepcopy(critic)

global discount = 0.9
global num_epochs = 2000
global steps_per_epoch = 200
s_current, a_current, rewards, s_next, a_next = sample_data(input_train, output_train, stage_costs)

for epoch in 1:num_epochs
    for step in 1:steps_per_epoch
        q_target = rewards .+ discount * vec(target_value_net(vcat(s_next, a_next)))
        grads = Flux.gradient(Flux.params(critic)) do
            q_values = vec(critic(vcat(s_current, a_current)))
            return Flux.mae(q_values, q_target)
        end
        Flux.Optimise.update!(ADAM(), Flux.params(critic), grads)
        soft_update!(target_value_net, critic)
    end
    println("Epoch completed: ", epoch) 
end

@save "critic_model.bson" critic