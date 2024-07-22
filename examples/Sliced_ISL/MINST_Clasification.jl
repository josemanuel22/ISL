using ISL
using Flux
using MLDatasets
using Images
using ImageTransformations  # For resizing images if necessary
using LinearAlgebra

# Load training data (images, labels)
x_train, y_train = MLDatasets.MNIST.traindata()
# Load test data (images, labels)
x_test, y_test = MLDatasets.MNIST.testdata()
# Convert grayscale to float
x_train = Float32.(x_train)
# Create labels batch
y_train = Flux.onehotbatch(y_train, 0:9)

# Define the model
model = Chain(Dense(784 + 1, 256, relu), Dense(256, 10, relu), softmax)
flattened_x_train = Flux.flatten(x_train)
flattened_y_train = Flux.flatten(y_train)
concatenated_data = vcat(flattened_x_train, flattened_y_train)

#train_data = [(Flux.flatten(x_train), Flux.flatten(y_train))]

batch_size = 1000
#train_loader = DataLoader(train_x; batchsize=batch_size, shuffle=false, partial=false)
loader_imgs = DataLoader(
    flattened_x_train; batchsize=batch_size, shuffle=true, partial=false
)

loader_class = DataLoader(
    flattened_y_train; batchsize=batch_size, shuffle=true, partial=false
)

noise_model = Normal(0.0f0, 1.0f0)
hparams = HyperParamsSlicedISL(;
    K=10, samples=1000, epochs=60, η=1e-2, noise_model=noise_model, m=5
)

total_loss = []
@showprogress for _ in 1:1
    append!(
        total_loss,
        sliced_invariant_statistical_loss_clasification(
            model, loader_imgs, loader_class, hparams
        ),
    )
end

concatenated_pairs = Float32[]
for i in 2:2:length(total_loss)
    append!(concatenated_pairs, total_loss[i])
end

acc = 0
for i in 1:100
    if argmax(model(vcat(flattened_x_train[:, i], randn()))) ==
        argmax(flattened_y_train[:, i])
        acc += 1
    end
end
acc / 100
