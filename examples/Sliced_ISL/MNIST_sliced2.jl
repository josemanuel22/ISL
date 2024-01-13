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
    return (reshape(Float32.(train_x), 28 * 28, :), train_y)#, (test_x, test_y)
end

(images, labels) = load_mnist()

n_outputs = length(unique(labels))

ys = [Flux.onehot(labels, 0:9) for labels in labels]

n_inputs, n_latent, n_outputs = 28 * 28, 50, 10
model = Chain(
    Dense(n_inputs, n_latent, identity),
    Dense(n_latent, n_latent, identity),
    Dense(n_latent, n_outputs, identity),
    softmax,
)
loss(x, y) = Flux.crossentropy(model(x), y)

function create_batch(r)
    xs = images[:, r]
    ys = [Flux.onehot(labels, 0:9) for labels in labels[r]]
    return (xs, Flux.batch(ys))
end

trainbatch = create_batch(1:5000)

opt = Flux.setup(Flux.Adam(hparams.η), model)
opt = ADAM()

@showprogress for _ in 1:1000
    Flux.train!(loss, Flux.params(model), [trainbatch], opt)
end

model(images[:, 1])
img2 = reshape(images[:, 1], 28, 28)
display(Gray.(img2))
