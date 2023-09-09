using Flux
using Random
using Statistics

using AdaptativeBlockLearning
using Distributions
using DataFrames
using CSV
using Plots

include("utils.jl")

# Hyperparameters and configuration of AR process
@Base.kwdef mutable struct HyperParamsTS
    seed::Int            = 72                       # Random seed
    # AR process parameters
    ϕ::Vector{Float32}   = [.7f0, .2f0, .1f0]      # AR coefficients (=> AR(3))
    proclen::Int         = 2000                      # Process length
    # Recurrent net parameters
    dev                  = cpu                      # Device: cpu or gpu
    opt                  = ADAM                     # Optimizer
    η::Float64           = 1e-3                     # Learning rate
    hidden_nodes::Int    = 10                       # Number of hidden nodes
    hidden_layers::Int   = 2                        # Number of hidden layers
    layer                = RNN                      # Type of layer, should be one of LSTM, GRU, RNN
    epochs::Int          = 100                      # Number of epochs
    seqlen::Int          = 49                       # Sequence length to use as input
    seqshift::Int        = 1                        # Shift between sequences (see utils.jl)
    train_ratio::Float64 = 0.5                       # Percentage of data in the train set
    verbose::Bool        = true                     # Whether we log the results during training or not
    noise_model          = Normal(0.0f0, 1.0f0)     # Noise to add to the data
    K                    = 5                        # Number of simulted observations
end

# Creates a model according to the pre-defined hyperparameters `args`
function build_model(args)
    Chain(
        args.layer(1, args.hidden_nodes),
        [args.layer(args.hidden_nodes, args.hidden_nodes) for _ ∈ 1:args.hidden_layers-1]...,
        Dense(args.hidden_nodes, 1, identity)
    ) |> args.dev
end

nn_model = Chain(
    RNN(1 => 32, relu),
    Dense(32 => 1, identity)
)

# Creates training and testing samples according to hyperparameters `args`
function generate_train_test_data(args)
    # Generate full AR process
    data = generate_process(args.ϕ, args.proclen)
    # Create input X and output y (series shifted by 1)
    X, y = data[1:end-1], data[2:end]
    # Split data into training and testing sets
    idx = round(Int, args.train_ratio * length(X))
    Xtrain, Xtest = X[1:idx], X[idx+1:end]
    ytrain, ytest = y[1:idx], y[idx+1:end]

    return (Xtrain, Xtest, ytrain, ytest)
    # Transform data to time series batches and return
    #map(x -> batch_timeseries(x, args.seqlen, args.seqshift) |> args.dev,
    #    (Xtrain, Xtest, ytrain, ytest))
end

function ts_adaptative_block_learning(nn_model, x, y, hparams)
    # Warm up recurrent model on first observation
    #nn_model(x[1])

    #@assert length(data) == hparams.samples

    losses = []
    optim = Flux.setup(Flux.Adam(hparams.η), nn_model)
    @showprogress for epoch in 1:(hparams.epochs)
        Flux.reset!(nn_model)
        loss, grads = Flux.withgradient(nn_model) do nn
            aₖ = zeros(hparams.K + 1)
            for yₜ in y
                xₖ = rand(hparams.noise_model, hparams.K)
                yₖ = nn(xₖ')
                aₖ += generate_aₖ(yₖ, yₜ)
            end
            scalar_diff(aₖ ./ sum(aₖ))
        end
        Flux.update!(optim, nn_model, grads[1])
        push!(losses, loss)
    end
    return losses
end

# Trains and outputs the model according to the chosen hyperparameters `args`
function train_model(hparams)
    Random.seed!(hparams.seed)
    # Create recurrent model
    nn_model = build_model(hparams)
    # Get data
    Xtrain, Xtest, ytrain, ytest = generate_train_test_data(hparams)

    X = [[x] for x ∈ Xtrain]

    loss = ts_adaptative_block_learning(nn_model, X, ytrain, hparams)
    return model
end

cd(@__DIR__)

hparams = HyperParamsTS()    # Set up hyperparameters
m = train_model(args)   # Train and output model

# Generate AR process

data =  CSV.read("/Users/jmfrutos/Desktop/HistoricalQuotes.csv", DataFrame, stripwhitespace=true)
price = tryparse.(Float32, replace.(data.Close,  "\$" => ""))
X = price[1:end-1]
Y = price[2:end]

X = [[x] for x ∈ X]

epochs = 100
opt = ADAM()
θ = Flux.params(nn_model) # Keep track of the model parameters
@showprogress for epoch ∈ 1:epochs # Training loop
    Flux.reset!(nn_model) # Reset the hidden state of the RNN
    # Compute the gradient of the mean squared error loss
    ∇ = gradient(θ) do
        nn_model(X[1]) # Warm-up the model
        sum(Flux.Losses.mse.([nn_model(x)[1] for x ∈ X[2:end]], Y[2:end]))
    end
    Flux.update!(opt, θ, ∇) # Update the parameters
end

loss = ts_adaptative_block_learning(nn_model, X, Y, hparams)
