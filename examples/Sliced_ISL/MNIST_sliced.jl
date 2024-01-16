using ISL
using Flux
using MLDatasets
using Images
using ImageTransformations  # For resizing images if necessary
using LinearAlgebra

function load_mnist()
    # Load MNIST data
    train_x, train_y = MLDatasets.MNIST.traindata()
    test_x, test_y = MLDatasets.MNIST.testdata()

    return (reshape(Float32.(train_x), 28 * 28, :)[:, 1:5000], train_y[1:5000])#, (test_x, test_y)
end

function load_mnist(digit::Int)
    # Load MNIST data
    train_x, train_y = MLDatasets.MNIST.traindata()
    test_x, test_y = MLDatasets.MNIST.testdata()

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

    return (reshape(Float32.(image_tensor), 28 * 28, :), train_y)
end

(train_x, train_y) = load_mnist()
(train_x, train_y) = (train_x[:, 1:5000], train_y[1:5000])
(train_x, train_y) = load_mnist(0)
(train_x, train_y) = load_mnist(9, 100)
(train_x, train_y) = load_mnist_normalized(8, 100)

# Dimension
dims = 100

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
    Dense(36, 1, 11),
    Dropout(0.2),
    softmax,
)

model = Chain(
    Dense(dims, 128, elu), Dense(128, 256, elu), Dense(256, 28 * 28, identity), Dropout(0.2)
)

model = Chain(
    Dense(dims, 256, relu),
    #BatchNorm(256),
    Dense(256, 512, relu),
    #BatchNorm(512, relu),
    Dense(512, 28 * 28, identity),
    x -> reshape(x, 28, 28, 1, :),
    Conv((3, 3), 1 => 16, relu),
    MaxPool((2, 2)),
    x -> reshape(x, :, size(x, 4)),
    Flux.flatten,
    Dense(2704, 28 * 28),
)

# weight initialization as given in the paper https://arxiv.org/abs/1511.06434
dcgan_init(shape...) = randn(Float32, shape...) * 0.02f0

function Discriminator()
    return Chain(
        Conv((4, 4), 1 => 64; stride=2, pad=1, init=dcgan_init),
        x -> leakyrelu.(x, 0.2f0),
        Dropout(0.25),
        Conv((4, 4), 64 => 128; stride=2, pad=1, init=dcgan_init),
        x -> leakyrelu.(x, 0.2f0),
        Dropout(0.25),
        x -> reshape(x, 7 * 7 * 128, :),
        Dense(7 * 7 * 128, 1),
    )
end

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
        Flux.flatten,
        x -> tanh.(x),
    )
end

model = gpu(Generator(latent_dim))
#model = Chain( ConvTranspose((7, 7), 100 => 256, stride=1, padding=0), BatchNorm(256, relu), ConvTranspose((4, 4), 256 => 128, stride=2, padding=1), BatchNorm(128, relu), ConvTranspose((4, 4), 128 => 1, stride=2, padding=1), tanh ))

# Mean vector (zero vector of length dim)
mean_vector = zeros(dims)

# Covariance matrix (identity matrix of size dim x dim)
cov_matrix = Diagonal(ones(dims))

# Create the multivariate normal distribution
noise_model = MvNormal(mean_vector, cov_matrix)

n_samples = 10000

hparams = gpu(
    HyperParamsSlicedISL(;
        K=10, samples=100, epochs=50, η=1e-2, noise_model=noise_model, m=100
    ),
)

# Create a data loader for training
batch_size = 100
train_loader = DataLoader(train_x; batchsize=batch_size, shuffle=false, partial=false)
train_loader = gpu(DataLoader(train_x; batchsize=batch_size, shuffle=false, partial=false))

total_loss = []
@showprogress for _ in 1:1
    append!(
        total_loss,
        sliced_invariant_statistical_loss_optimized_4(model, train_loader, hparams),
    )
end

img = model(Float32.(rand(hparams.noise_model, 1)))
img2 = reshape(img, 28, 28)
#display(Gray.(img2))
MLDatasets.MNIST.convert2image(Gray.(img2))
transformed_matrix = Float32.(img2 .> 0.4)
display(Gray.(transformed_matrix))

using Base.Iterators: partition
# Load MNIST dataset
images = MLDatasets.MNIST(:train).features
# Normalize to [-1, 1]
image_tensor = reshape(@.(2.0f0 * images - 1.0f0), 28, 28, :)
# Partition into batches
#data = [image_tensor[:, :, r] |> gpu for r in partition(1:60000, 100)]

train_data = reshape(image_tensor, 28 * 28, :)

train_loader = DataLoader(train_data; batchsize=batch_size, shuffle=true, partial=false)
