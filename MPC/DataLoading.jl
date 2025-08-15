using Statistics
using DelimitedFiles
using Distributions
using LinearAlgebra

# Loading Datasets
FR_price = readdlm("Data/FR_Price.csv", ',', Float64)
FR_price_orig = vcat(FR_price'...)
FR_price = FR_price_orig.+1e-4

grid_price = readdlm("Data/Grid_Price.csv", ',', Float64) 

Files = [
    "Data/01.csv", "Data/02.csv", "Data/03.csv", 
    "Data/04.csv", "Data/05.csv", "Data/06.csv",
    "Data/07.csv", "Data/08.csv", "Data/09.csv", 
    "Data/10.csv", "Data/11.csv", "Data/12.csv"
]

global signal = []

for i in 1:12
    filename = Files[i]
    originalsignal = readdlm(filename, ',', Float64)[1:(end-1),:]   #positive:charging / negative: discharging
    if i == 1
        global signal = vcat(originalsignal...)
    else
        global signal = [signal; vcat(originalsignal...)]
    end
    if i == 12
       global signal = [signal; originalsignal[end,end]]
    end
end
signal = min.(max.(signal, -1),1)

grid_price = grid_price[1:364, :]'

Xn_price = reshape(grid_price, (:, 52)) 
p, n = size(Xn_price)
Sn = Xn_price * Xn_price' / n
In = Diagonal(ones(p))

m_n = tr(Sn * In) / p
d_n2 = norm(Sn - m_n * In)^2
b_nbar = (1/n^2)*sum(norm(Xn_price[:, i] * Xn_price[:, i]' - Sn)^2 for i = 1:n)
b_n2 = min(b_nbar, d_n2)
a_n2 = d_n2 - b_n2

Sigma_n = (b_n2 / d_n2) * m_n * In + (a_n2 / d_n2) * Sn
mu_n = mean(Xn_price, dims = 2)
mu_n = dropdims(mu_n, dims = 2)
d_grid_price = MvNormal(mu_n, Sigma_n)

FR_price_mean = mean(log.(FR_price))
FR_price_var = std(log.(FR_price))

d_FR_price = LogNormal(FR_price_mean, FR_price_var)

max_grid_price = maximum(vcat(grid_price...))
min_grid_price = minimum(vcat(grid_price...))
max_FR_price = maximum(FR_price_orig)
min_FR_price = minimum(FR_price_orig)

signal_index = collect(1:1800:(size(signal)[1]-1800))


