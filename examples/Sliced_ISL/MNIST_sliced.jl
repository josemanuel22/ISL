using ISL
using Flux
using MLDatasets
using Images
using ImageTransformations  # For resizing images if necessary

function load_mnist(digit::Int)
    # Load MNIST data
    train_x, train_y = MLDatasets.MNIST.traindata()
    test_x, test_y = MLDatasets.MNIST.testdata()

    # Find indices where the label is digit
    selected_indices = findall(x -> x == digit, train_y)

    selected_images = train_x[:, :, selected_indices]

    return (reshape(Float32.(selected_images), 784, :), train_y)#, (test_x, test_y)
end

(train_x, train_y) = load_mnist(5)


model = Chain(
    Dense(3, 512, relu),
    Dense(512, 28*28, sigmoid)
)

model = Chain(
    Dense(3, 256, relu),
    #BatchNorm(256),
    Dense(256, 512, relu),
    #BatchNorm(512, relu),
    Dense(512, 28*28, identity),
    x -> reshape(x, 28, 28, 1, :),
    Conv((3, 3), 1=>16, relu),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x, 4)),
    Flux.flatten,
    Dense(2704, 28*28)
)

# Define hyperparameters
noise_model = MvNormal([0.0, 0.0, 0.0], [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])
n_samples = 10000

hparams = HyperParamsSlicedISL(;
    K=10, samples=1000, epochs=5, η=1e-2, noise_model=noise_model, m=20
)

# Create a data loader for training
batch_size = 1000
train_loader = DataLoader(train_x; batchsize=batch_size, shuffle=false, partial=false)

total_loss = []
for _ in 1:10
    append!(total_loss, sliced_invariant_statistical_loss(model, train_loader, hparams))
end

img = model(Float32.(rand(hparams.noise_model, 1)))
img2 = reshape(img, 28, 28)
display(Gray.(img2))
transformed_matrix = Float32.(img2 .> 0.1)
display(Gray.(transformed_matrix))
