var documenterSearchIndex = {"docs":
[{"location":"DeepAR/#DeepAR-Module-Documentation","page":"DeepAR","title":"DeepAR Module Documentation","text":"","category":"section"},{"location":"DeepAR/#Overview","page":"DeepAR","title":"Overview","text":"","category":"section"},{"location":"DeepAR/","page":"DeepAR","title":"DeepAR","text":"This module implements the DeepAR model, a method for probabilistic forecasting with autoregressive recurrent networks. DeepAR is designed to model time series data with complex patterns and provide accurate probabilistic forecasts. This implementation is inspired by the approach described in the DeepAR paper and is adapted for use in Julia with Flux and StatsBase for neural network modeling and statistical operations, respectively. This module has subsequently been extracted into a separate repository, see https://github.com/josemanuel22/DeepAR.jl","category":"page"},{"location":"DeepAR/#Installation","page":"DeepAR","title":"Installation","text":"","category":"section"},{"location":"DeepAR/","page":"DeepAR","title":"DeepAR","text":"Before using this module, ensure that you have installed the required Julia packages: Flux, StatsBase, and Random. You can add these packages to your Julia environment by running:","category":"page"},{"location":"DeepAR/","page":"DeepAR","title":"DeepAR","text":"using Pkg\nPkg.add([\"Flux\", \"StatsBase\", \"Random\"])","category":"page"},{"location":"DeepAR/#Module-Components","page":"DeepAR","title":"Module Components","text":"","category":"section"},{"location":"DeepAR/#DeepArParams-Struct","page":"DeepAR","title":"DeepArParams Struct","text":"","category":"section"},{"location":"DeepAR/","page":"DeepAR","title":"DeepAR","text":"A structure to hold hyperparameters for the DeepAR model.","category":"page"},{"location":"DeepAR/","page":"DeepAR","title":"DeepAR","text":"Base.@kwdef mutable struct DeepArParams\n    η::Float64 = 1e-2    # Learning rate\n    epochs::Int = 10     # Number of training epochs\n    n_mean::Int = 100    # Number of samples for predictive mean\nend","category":"page"},{"location":"DeepAR/#train_DeepAR-Function","page":"DeepAR","title":"train_DeepAR Function","text":"","category":"section"},{"location":"DeepAR/","page":"DeepAR","title":"DeepAR","text":"Function to train a DeepAR model.","category":"page"},{"location":"DeepAR/","page":"DeepAR","title":"DeepAR","text":"train_DeepAR(model, loaderXtrain, loaderYtrain, hparams) -> Vector{Float64}","category":"page"},{"location":"DeepAR/","page":"DeepAR","title":"DeepAR","text":"Arguments:\nmodel: The DeepAR model to be trained.\nloaderXtrain: DataLoader containing input sequences for training.\nloaderYtrain: DataLoader containing target sequences for training.\nhparams: An instance of DeepArParams specifying training hyperparameters.\nReturns: A vector of loss values recorded during training.","category":"page"},{"location":"DeepAR/#forecasting_DeepAR-Function","page":"DeepAR","title":"forecasting_DeepAR Function","text":"","category":"section"},{"location":"DeepAR/","page":"DeepAR","title":"DeepAR","text":"Function to generate forecasts using a trained DeepAR model.","category":"page"},{"location":"DeepAR/","page":"DeepAR","title":"DeepAR","text":"forecasting_DeepAR(model, ts, t₀, τ; n_samples=100) -> Vector{Float32}","category":"page"},{"location":"DeepAR/","page":"DeepAR","title":"DeepAR","text":"Arguments:\nmodel: The trained DeepAR model.\nts: Time series data for forecasting.\nt₀: Starting time step for forecasting.\nτ: Number of time steps to forecast.\nn_samples: Number of samples to draw for each forecast step (default: 100).\nReturns: A vector containing the forecasted values for each time step.","category":"page"},{"location":"DeepAR/#Example-Usage","page":"DeepAR","title":"Example Usage","text":"","category":"section"},{"location":"DeepAR/","page":"DeepAR","title":"DeepAR","text":"This section demonstrates how to use the DeepAR model for probabilistic forecasting of time series data.","category":"page"},{"location":"DeepAR/","page":"DeepAR","title":"DeepAR","text":"# Define AR model parameters and generate training and testing data\nar_hparams = ARParams(...)\nloaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(...)\n\n# Initialize the DeepAR model\nmodel = Chain(...)\n\n# Define hyperparameters and train the model\ndeepar_params = DeepArParams(...)\nlosses = train_DeepAR(model, loaderXtrain, loaderYtrain, deepar_params)\n\n# Perform forecasting\nt₀, τ = 100, 20\npredictions = forecasting_DeepAR(model, collect(loaderXtrain)[1], t₀, τ; n_samples=100)","category":"page"},{"location":"DeepAR/#References","page":"DeepAR","title":"References","text":"","category":"section"},{"location":"DeepAR/","page":"DeepAR","title":"DeepAR","text":"\"DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks\" by David Salinas, Valentin Flunkert, and Jan Gasthaus.","category":"page"},{"location":"Examples/#Quick-Start-Guide","page":"Example","title":"Quick Start Guide","text":"","category":"section"},{"location":"Examples/","page":"Example","title":"Example","text":"After installing the package, you can immediately start experimenting with the examples provided in the examples/ directory. These examples are designed to help you understand how to apply the package for different statistical learning tasks, including learning 1-D distributions and time series prediction.","category":"page"},{"location":"Examples/#Learning-1-D-distributions","page":"Example","title":"Learning 1-D distributions","text":"","category":"section"},{"location":"Examples/","page":"Example","title":"Example","text":"The following example demonstrates how to learn a 1-D distribution from a benchmark dataset. It utilizes a simple neural network architecture for both the generator and the discriminator within the context of an Invariant Statistical Learning (ISL) framework.","category":"page"},{"location":"Examples/","page":"Example","title":"Example","text":"# Example file: examples/Learning1d_distribution/benchmark_unimodal.jl\n\nusing ISL # Import the ISL module\ninclude(\"../utils.jl\")  # Include necessary utilities\n\n@test_experiments \"N(0,1) to N(23,1)\" begin\n    # Define the generator and discriminator networks with ELU activation\n    gen = Chain(Dense(1, 7), elu, Dense(7, 13), elu, Dense(13, 7), elu, Dense(7, 1))\n\n    # Set up the noise and target models\n    noise_model = Normal(0.0f0, 1.0f0)\n    target_model = Normal(4.0f0, 2.0f0)\n    n_samples = 10000   \n\n    # Configure the hyperparameters for the ISL\n    hparams = AutoISLParams(;\n        max_k=10, samples=1000, epochs=1000, η=1e-2, transform=noise_model\n    )\n\n    # Prepare the dataset and the loader for training\n    train_set = Float32.(rand(target_model, hparams.samples))\n    loader = Flux.DataLoader(train_set; batchsize=-1, shuffle=true, partial=false)\n\n    # Train the model using the ISL\n    auto_invariant_statistical_loss(gen, loader, hparams)\n\n    # Visualize the learning results\n    plot_global(\n        x -> quantile.(-target_model, cdf(noise_model, x)),\n        noise_model,\n        target_model,\n        gen,\n        n_samples,\n        (-3:0.1:3),\n        (-2:0.1:10),\n    )\nend","category":"page"},{"location":"Examples/","page":"Example","title":"Example","text":"(Image: Example Image)","category":"page"},{"location":"Examples/#Time-Series","page":"Example","title":"Time Series","text":"","category":"section"},{"location":"Examples/","page":"Example","title":"Example","text":"This section showcases two examples: predicting a univariate time series with an AutoRegressive model and forecasting electricity consumption. These examples illustrate the application of ISL for time series analysis.","category":"page"},{"location":"Examples/","page":"Example","title":"Example","text":"Example 1: AutoRegressive Model Prediction","category":"page"},{"location":"Examples/","page":"Example","title":"Example","text":"@test_experiments \"testing AutoRegressive Model 1\" begin\n    # --- Model Parameters and Data Generation ---\n\n    # Define AR model parameters\n    ar_hparams = ARParams(;\n        ϕ=[0.5f0, 0.3f0, 0.2f0],  # Autoregressive coefficients\n        x₁=rand(Normal(0.0f0, 1.0f0)),  # Initial value from a Normal distribution\n        proclen=2000,  # Length of the process\n        noise=Normal(0.0f0, 0.2f0),  # Noise in the AR process\n    )\n\n    # Define the recurrent and generative models\n    recurrent_model = Chain(RNN(1 => 10, relu), RNN(10 => 10, relu))\n    generative_model = Chain(Dense(11, 16, relu), Dense(16, 1, identity))\n\n    # Generate training and testing data\n    n_series = 200  # Number of series to generate\n    loaderXtrain, loaderYtrain, loaderXtest, loaderYtest = generate_batch_train_test_data(\n        n_series, ar_hparams\n    )\n\n    # --- Training Configuration ---\n\n    # Define hyperparameters for time series prediction\n    ts_hparams = HyperParamsTS(;\n        seed=1234,\n        η=1e-3,  # Learning rate\n        epochs=n_series,\n        window_size=1000,  # Size of the window for prediction\n        K=10,  # Hyperparameter K (if it has a specific use, add a comment)\n    )\n\n    # Train model and calculate loss\n    loss = ts_invariant_statistical_loss_one_step_prediction(\n        recurrent_model, generative_model, loaderXtrain, loaderYtrain, ts_hparams\n    )\n\n    # --- Visualization ---\n\n    # Plotting the time series prediction\n    plot_univariate_ts_prediction(\n        recurrent_model,\n        generative_model,\n        collect(loaderXtrain)[2],  # Extract the first batch for plotting\n        collect(loaderXtest)[2],  # Extract the first batch for plotting\n        ts_hparams;\n        n_average=1000,  # Number of predictions to average\n    )\nend","category":"page"},{"location":"Examples/","page":"Example","title":"Example","text":"(Image: Example Image)","category":"page"},{"location":"Examples/","page":"Example","title":"Example","text":"Example 2: Electricity Consumption Forecasting","category":"page"},{"location":"Examples/","page":"Example","title":"Example","text":"@test_experiments \"testing electricity-c\" begin\n     # Data loading and preprocessing\n    csv_file_path = \"examples/time_series_predictions/data/LD2011_2014.txt\"\n\n    # Specify custom types for certain columns to ensure proper data handling\n    column_types = Dict(\n        \"MT_005\" => Float32,\n        \"MT_006\" => Float32,\n        \"MT_007\" => Float32,\n        \"MT_008\" => Float32,\n        \"MT_168\" => Float32,\n        \"MT_333\" => Float32,\n        \"MT_334\" => Float32,\n        \"MT_335\" => Float32,\n        \"MT_336\" => Float32,\n        \"MT_338\" => Float32,\n    )\n\n    df = DataFrame(\n        CSV.File(csv_file_path; delim=';', header=true, decimal=',', types=column_types)\n    )\n\n    # Hyperparameters setup\n    hparams = HyperParamsTS(; seed=1234, η=1e-2, epochs=2000, window_size=1000, K=10)\n\n    # Model definition\n    rec = Chain(RNN(1 => 3, relu), RNN(3 => 3, relu))\n    gen = Chain(Dense(4, 10, identity), Dense(10, 1, identity))\n\n    # Training and testing data setup\n    start, num_training_data, num_test_data = 35040, 35040, 1000\n\n    # Aggregate time series data for training and testing\n    selected_columns = [\n        \"MT_005\",\n        \"MT_006\",\n        \"MT_007\",\n        \"MT_008\",\n        \"MT_168\",\n        \"MT_333\",\n        \"MT_334\",\n        \"MT_335\",\n        \"MT_336\",\n        \"MT_338\",\n    ]\n\n    loader_xtrain, loader_ytrain, loader_xtest = aggregate_time_series_data(\n        df, selected_columns, start, num_training_data, num_test_data\n    )\n\n    # Model training\n    losses = []\n    @showprogress for _ in 1:100\n        loss = ts_invariant_statistical_loss(\n            rec, gen, loader_xtrain, loader_ytrain, hparams\n        )\n        append!(losses, loss)\n    end\n\n    # Forecasting\n    τ = 24\n    xtrain = collect(loaderXtrain)[1]\n    xtest = collect(loaderXtest)[1]\n    prediction, stds = ts_forecast(\n        rec, gen, xtrain, xtest, τ; n_average=1000, noise_model=Normal(0.0f0, 1.0f0)\n    )\n    plot(prediction[1:τ])\n    plot!(xtest[1:τ])\nend","category":"page"},{"location":"Examples/","page":"Example","title":"Example","text":"Prediction τ = 24 (1 day)","category":"page"},{"location":"Examples/","page":"Example","title":"Example","text":"(Image: Example Image)","category":"page"},{"location":"Examples/","page":"Example","title":"Example","text":"Prediction τ = 168 (7 days)","category":"page"},{"location":"Examples/","page":"Example","title":"Example","text":"(Image: Example Image)","category":"page"},{"location":"Gans/#Generative-Adversarial-Networks-(GANs)-Module-Overview","page":"GANs","title":"Generative Adversarial Networks (GANs) Module Overview","text":"","category":"section"},{"location":"Gans/","page":"GANs","title":"GANs","text":"This repository includes a dedicated folder that contains implementations of different Generative Adversarial Networks (GANs), showcasing a variety of approaches within the GAN framework. Our collection includes:","category":"page"},{"location":"Gans/","page":"GANs","title":"GANs","text":"Vanilla GAN: Based on the foundational GAN concept introduced in \"Generative Adversarial Nets\" by Goodfellow et al. This implementation adapts and modifies the code from FluxGAN repository to fit our testing needs.\nWGAN (Wasserstein GAN): Implements the Wasserstein GAN as described in \"Wasserstein GAN\" by Arjovsky et al., providing an advanced solution to the issue of training stability in GANs. Similar to Vanilla GAN, we have utilized and slightly adjusted the implementation from the FluxGAN repository.\nMMD-GAN (Maximum Mean Discrepancy GAN): Our implementation of MMD-GAN is inspired by the paper \"MMD GAN: Towards Deeper Understanding of Moment Matching Network\" by Li et al. Unlike the previous models, the MMD-GAN implementation has been rewritten in Julia, transitioning from the original Python code provided by the authors.","category":"page"},{"location":"Gans/#Objective","page":"GANs","title":"Objective","text":"","category":"section"},{"location":"Gans/","page":"GANs","title":"GANs","text":"The primary goal of incorporating these GAN models into our repository is to evaluate the effectiveness of ISL (Invariant Statistical Learning) methods as regularizers for GAN-based solutions. Specifically, we aim to address the challenges presented in the \"Helvetica scenario,\" exploring how ISL methods can enhance the robustness and generalization of GANs in generating high-quality synthetic data.","category":"page"},{"location":"Gans/#Implementation-Details","page":"GANs","title":"Implementation Details","text":"","category":"section"},{"location":"Gans/","page":"GANs","title":"GANs","text":"For each GAN variant mentioned above, we have made certain adaptations to the original implementations to ensure compatibility with our testing framework and the objectives of the ISL method integration. These modifications range from architectural adjustments to the optimization process, aiming to optimize the performance and efficacy of the ISL regularizers within the GAN context.","category":"page"},{"location":"Gans/","page":"GANs","title":"GANs","text":"We encourage interested researchers and practitioners to explore the implementations and consider the potential of ISL methods in improving GAN architectures. For more detailed insights into the modifications and specific implementation choices, please refer to the code and accompanying documentation within the respective folders for each GAN variant.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = ISL","category":"page"},{"location":"#ISL.jl-Documentation-Guide","page":"Home","title":"ISL.jl Documentation Guide","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Welcome to the documentation for ISL.jl, a Julia package designed for Invariant Statistical Learning. This guide provides a systematic overview of the modules, constants, types, and functions available in ISL.jl. Our documentation aims to help you quickly find the information you need to effectively utilize the package.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [ISL]\nOrder   = [:module, :constant, :type]","category":"page"},{"location":"#ISL.AutoISLParams","page":"Home","title":"ISL.AutoISLParams","text":"AutoISLParams\n\nHyperparameters for the method invariant_statistical_loss\n\n@with_kw struct AutoISLParams\n    samples::Int64 = 1000\n    epochs::Int64 = 100\n    η::Float64 = 1e-3\n    max_k::Int64 = 10\n    transform = Normal(0.0f0, 1.0f0)\nend;\n\n\n\n\n\n","category":"type"},{"location":"#ISL.HyperParamsTS","page":"Home","title":"ISL.HyperParamsTS","text":"HyperParamsTS\n\nHyperparameters for the method ts_adaptative_block_learning\n\n\n\n\n\n","category":"type"},{"location":"#ISL.ISLParams","page":"Home","title":"ISL.ISLParams","text":"ISLParams\n\nHyperparameters for the method adaptative_block_learning\n\n@with_kw struct ISLParams\n    samples::Int64 = 1000               # number of samples per histogram\n    K::Int64 = 2                        # number of simulted observations\n    epochs::Int64 = 100                 # number of epochs\n    η::Float64 = 1e-3                   # learning rate\n    transform = Normal(0.0f0, 1.0f0)    # transform to apply to the data\nend;\n\n\n\n\n\n","category":"type"},{"location":"","page":"Home","title":"Home","text":"invariant_statistical_loss\nauto_invariant_statistical_loss\nts_invariant_statistical_loss_one_step_prediction\nts_invariant_statistical_loss\nts_invariant_statistical_loss_multivariate","category":"page"},{"location":"#ISL.invariant_statistical_loss","page":"Home","title":"ISL.invariant_statistical_loss","text":"invariant_statistical_loss(model, data, hparams)\n\nCustom loss function for the model. model is a Flux neuronal network model, data is a loader Flux object and hparams is a HyperParams object.\n\nArguments\n\nnn_model::Flux.Chain: is a Flux neuronal network model\ndata::Flux.DataLoader: is a loader Flux object\nhparams::HyperParams: is a HyperParams object\n\n\n\n\n\n","category":"function"},{"location":"#ISL.auto_invariant_statistical_loss","page":"Home","title":"ISL.auto_invariant_statistical_loss","text":"auto_invariant_statistical_loss(model, data, hparams)\n\nCustom loss function for the model.\n\nThis method gradually adapts K (starting from 2) up to max_k (inclusive). The value of K is chosen based on a simple two-sample test between the histogram associated with the obtained result and the uniform distribution.\n\nTo see the value of K used in the test, set the logger level to debug before executing.\n\nArguments\n\nmodel::Flux.Chain: is a Flux neuronal network model\ndata::Flux.DataLoader: is a loader Flux object\nhparams::AutoAdaptativeHyperParams: is a AutoAdaptativeHyperParams object\n\n\n\n\n\n","category":"function"},{"location":"#ISL.ts_invariant_statistical_loss_one_step_prediction","page":"Home","title":"ISL.ts_invariant_statistical_loss_one_step_prediction","text":"ts_invariant_statistical_loss_one_step_prediction(rec, gen, Xₜ, Xₜ₊₁, hparams) -> losses\n\nCompute the loss for one-step-ahead predictions in a time series using a recurrent model and a generative model.\n\nArguments\n\nrec: The recurrent model that processes the input time series data Xₜ to generate a hidden state.\ngen: The generative model that, based on the hidden state produced by rec, predicts the next time step in the series.\nXₜ: Input time series data at time t, used as input to the recurrent model.\nXₜ₊₁: Actual time series data at time t+1, used for calculating the prediction loss.\nhparams: A struct of hyperparameters for the training process, which includes:\nη: Learning rate for the optimizers.\nK: The number of noise samples to generate for prediction.\nwindow_size: The segment length of the time series data to process in each training iteration.\nnoise_model: The model to generate noise samples for the prediction process.\n\nReturns\n\nlosses: A list of loss values computed for each iteration over the batches of data.\n\nDescription\n\nThis function iterates over batches of time series data, utilizing a sliding window approach determined by hparams.window_size to process segments of the series. In each iteration, it computes a hidden state using the recurrent model rec, generates predictions for the next time step with the generative model gen based on noise samples and the hidden state, and calculates the loss based on these predictions and the actual data Xₜ₊₁. The function updates both models using the Adam optimizer with gradients derived from the loss.\n\nExample\n\n# Define your recurrent and generative models\nrec = Chain(RNN(1 => 3, relu), RNN(3 => 3, relu))\ngen = Chain(Dense(4, 10, identity), Dense(10, 1, identity))\n\n# Prepare your time series data Xₜ and Xₜ₊₁\nXₜ = ...\nXₜ₊₁ = ...\n\n# Set up hyperparameters\nhparams = HyperParamsTS(; seed=1234, η=1e-2, epochs=2000, window_size=1000, K=10)\n\n# Compute the losses\nlosses = ts_invariant_statistical_loss_one_step_prediction(rec, gen, Xₜ, Xₜ₊₁, hparams)\n\n\n\n\n\n","category":"function"},{"location":"#ISL.ts_invariant_statistical_loss","page":"Home","title":"ISL.ts_invariant_statistical_loss","text":"ts_invariant_statistical_loss(rec, gen, Xₜ, Xₜ₊₁, hparams)\n\nTrain a model for time series data with statistical invariance loss method.\n\nArguments\n\nrec: The recurrent neural network (RNN) responsible for encoding the time series data.\ngen: The generative model used for generating future time series data.\nXₜ: An array of input time series data at time t.\nXₜ₊₁: An array of target time series data at time t+1.\nhparams::NamedTuple: A structure containing hyperparameters for training. It should include the following fields:\nη::Float64: Learning rate for optimization.\nwindow_size::Int: Size of the sliding window used during training.\nK::Int: Number of samples in the generative model.\nnoise_model: Noise model used for generating random noise.\n\nReturns\n\nlosses::Vector{Float64}: A vector containing the training loss values for each iteration.\n\nDescription\n\nThis function train a model for time series data with statistical invariance loss method. It utilizes a recurrent neural network (rec) to encode the time series data at time t and a generative model (gen) to generate future time series data at time t+1. The training process involves optimizing both the rec and gen models.\n\nThe function iterates through the provided time series data (Xₜ and Xₜ₊₁) in batches, with a sliding window of size window_size.\n\n\n\n\n\n","category":"function"},{"location":"#ISL.ts_invariant_statistical_loss_multivariate","page":"Home","title":"ISL.ts_invariant_statistical_loss_multivariate","text":"ts_invariant_statistical_loss_multivariate(rec, gen, Xₜ, Xₜ₊₁, hparams) -> losses\n\nCalculate the time series invariant statistical loss for multivariate data using recurrent and generative models.\n\nArguments\n\nrec: The recurrent model to process input time series data Xₜ.\ngen: The generative model that works in conjunction with rec to generate the next time step predictions.\nXₜ: The input time series data at time t.\nXₜ₊₁: The actual time series data at time t+1 for loss calculation.\nhparams: A struct containing hyperparameters for the model. Expected fields include:\nη: Learning rate for the Adam optimizer.\nK: The number of samples to draw from the noise model.\nwindow_size: The size of the window to process the time series data in chunks.\nnoise_model: The statistical model to generate noise samples for the generative model.\n\nReturns\n\nlosses: An array containing the loss values computed for each batch in the dataset.\n\nDescription\n\nThis function iterates over the provided time series data Xₜ and Xₜ₊₁, processing each batch through the recurrent model rec to generate a state s, which is then used along with samples from noise_model to generate predictions with gen. The loss is calculated based on the difference between the generated predictions and the actual data Xₜ₊₁, and the models are updated using the Adam optimizer.\n\nExample\n\n# Define your recurrent and generative models here\nrec = Chain(RNN(1 => 3, relu), RNN(3 => 3, relu))\ngen = Chain(Dense(4, 10, identity), Dense(10, 1, identity))\n\n# Load or define your time series data Xₜ and Xₜ₊₁\nXₜ = ...\nXₜ₊₁ = ...\n\n# Define hyperparameters\nhparams = HyperParamsTS(; seed=1234, η=1e-2, epochs=2000, window_size=1000, K=10)\n\n# Calculate the losses\nlosses = ts_invariant_statistical_loss_multivariate(rec, gen, Xₜ, Xₜ₊₁, hparams)\n\n\n\n\n\n","category":"function"}]
}
