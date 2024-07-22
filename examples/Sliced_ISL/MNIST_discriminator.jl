using Flux
using Flux: onehotbatch, onecold, crossentropy, throttle
using MLDatasets
using Base.Iterators: partition
using Statistics: mean
using Zygote: gradient

# Load the data
train_x, train_y = MLDatasets.MNIST.traindata()
test_x, test_y = MLDatasets.MNIST.testdata()

train_x, train_y, test_x, test_y = load_data(hparams)
# Preprocess the data
train_x = Flux.flatten(train_x)
test_x = Flux.flatten(test_x)

# One-hot encode the labels
train_y = onehotbatch(train_y, 0:9)
test_y = onehotbatch(test_y, 0:9)

# Define the model
model = Chain(Dense(28^2, 128, relu), Dense(128, 10), softmax)

# Define the loss function
loss(x, y) = crossentropy(model(x), y)

# Setup the optimizer
opt = ADAM()

# Training function
function train_epoch!(data, model, loss, opt)
    for (x, y) in data
        gs = gradient(() -> loss(x, y), Flux.params(model))
        Flux.Optimise.update!(opt, Flux.params(model), gs)
    end
end

function load_data()
    # Load MNIST dataset
    train_x, train_y = MLDatasets.MNIST.traindata(Float32)
    test_x, test_y = MLDatasets.MNIST.testdata(Float32)
    # Normalize to [-1, 1] and convert it to WHCN
    train_x = reshape(@.(2.0f0 * train_x - 1.0f0), 28, 28, :)
    test_x = reshape(@.(2.0f0 * test_x - 1.0f0), 28, 28, :)
    #image_tensor = reshape(image_tensor, :, size(image_tensor, 4))
    # Partition into batches
    #data = [gpu(image_tensor[:, r]) for r in partition(1:60000, hparams.batch_size)]
    #return image_tensor
    return (
        reshape(Float32.(train_x), 28 * 28, :),
        train_y,
        reshape(Float32.(test_x), 28 * 28, :),
        test_y,
    )
end

# Prepare data loader

hparams = HyperParams()

#train_data = [(train_x[:, i], train_y[:, i]) for i in 1:size(train_x, 2)]
#test_data = [(test_x[:, i], test_y[:, i]) for i in 1:size(test_x, 2)]

# Training loop
epochs = 5
for epoch in 1:epochs
    @info "Epoch $epoch"
    train_epoch!(train_data, model, loss, opt)
    train_loss = mean(loss(train_x, train_y))
    test_loss = mean(loss(test_x, test_y))
    @info "Train loss: $train_loss, Test loss: $test_loss"
end

# Evaluate the model
preds = onecold(model(test_x))
accuracy = mean(preds .== onecold(test_y))
@info "Test accuracy: $accuracy"
