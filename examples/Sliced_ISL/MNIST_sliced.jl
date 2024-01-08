using ISL
using Flux
using MLDatasets
using Images
using ImageTransformations  # For resizing images if necessary
using LinearAlgebra

function load_mnist(digit::Int)
    # Load MNIST data
    train_x, train_y = MLDatasets.MNIST.traindata()

    # Find indices where the label is digit
    selected_indices = findall(x -> x == digit, train_y)

    selected_images = train_x[:, :, selected_indices]

    return (reshape(Float32.(selected_images), 28 * 28, :), train_y)#, (test_x, test_y)
end

function load_mnist(digit::Int, max::Int)
    # Load MNIST data
    train_x, train_y = MLDatasets.MNIST.traindata()

    # Find indices where the label is digit
    selected_indices = findall(x -> x == digit, train_y)

    selected_images = train_x[:, :, selected_indices[1:max]]

    return (reshape(Float32.(selected_images), 28 * 28, :), train_y)#, (test_x, test_y)
end

function load_mnist_normalized(digit::Int, max::Int)
    # Load MNIST data
    train_x, train_y = MLDatasets.MNIST.traindata()

    # Find indices where the label is digit
    selected_indices = findall(x -> x == digit, train_y)

    selected_images = train_x[:, :, selected_indices[1:max]]

    image_tensor = reshape(@.(2.0f0 * selected_images - 1.0f0), 28, 28, :)

    train_data = reshape(image_tensor, 28 * 28, :)

    return (train_data, train_y)
end

(train_x, train_y) = load_mnist_normalized(9, 200)

# Dimension
dims = 100
model = Chain(
    Dense(dims, 128),
    BatchNorm(128, elu),
    x -> reshape(x, 4, 4, 8, :),
    ConvTranspose((2, 2), 8 => 4),
    BatchNorm(4, elu),
    ConvTranspose((2, 2), 4 => 1),
    BatchNorm(1, elu),
    Flux.flatten,
    Dense(36, 28 * 28, identity),
    Dropout(0.2),
)

model = Chain(
    Dense(dims, 256, relu),
    Dense(256, 28 * 28, identity),
    x -> reshape(x, 28, 28, 1, :),
    Conv((3, 3), 1 => 8, relu),
    MaxPool((2, 2)),
    x -> reshape(x, :, size(x, 4)),
    Flux.flatten,
    Dense(1352, 28 * 28),
)

latent_dim = 100
# weight initialization as given in the paper https://arxiv.org/abs/1511.06434
dcgan_init(shape...) = randn(Float32, shape...) * 0.02f0
function Generator(latent_dim::Int)
    return Chain(
        Dense(latent_dim, 7 * 7 * 256),
        BatchNorm(7 * 7 * 256, relu),
        x -> reshape(x, 7, 7, 256, :),
        ConvTranspose((5, 5), 256 => 128; stride=1, pad=2, init=dcgan_init),
        BatchNorm(128, relu),
        ConvTranspose((4, 4), 128 => 64; stride=2, pad=1, init=dcgan_init),
        BatchNorm(64, relu),
        ConvTranspose((4, 4), 64 => 1; stride=2, pad=1, init=dcgan_init),
        x -> tanh.(x),
    )
end

model = Generator(latent_dim)

# Define hyperparameters

# Mean vector (zero vector of length dim)
mean_vector = zeros(dims)

# Covariance matrix (identity matrix of size dim x dim)
cov_matrix = Diagonal(ones(dims))

# Create the multivariate normal distribution
noise_model = MvNormal(mean_vector, cov_matrix)

#noise_model = MvNormal([0.0, 0.0, 0.0, 0.0], [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0])
n_samples = 10000

hparams = HyperParamsSlicedISL(;
    K=10, samples=128, epochs=468, η=1e-2, noise_model=noise_model, m=100
)

# Create a data loader for training
batch_size = 200
train_loader = DataLoader(train_x; batchsize=batch_size, shuffle=false, partial=false)

# Load MNIST dataset
images = MLDatasets.MNIST(:train).features
# Normalize to [-1, 1]
image_tensor = reshape(@.(2.0f0 * images - 1.0f0), 28 * 28, :)
train_loader = DataLoader(image_tensor; batchsize=128, shuffle=false, partial=false)

total_loss = []
@showprogress for _ in 1:1
    append!(
        total_loss,
        sliced_invariant_statistical_loss_10_images(model, train_loader, hparams),
    )
end

img = model(Float32.(rand(hparams.noise_model, 1)))
img2 = reshape(img, 28, 28)
display(Gray.(img2))

μ = mean(img)
σ = var(img)
img = (img .- μ) ./ σ
img2 = reshape(img, 28, 28)
Gray.(img2 / 5)

transformed_matrix = Float32.(img2 .> 0.22)
display(Gray.(transformed_matrix))
