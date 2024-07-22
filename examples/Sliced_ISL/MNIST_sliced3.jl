using ISL
using Flux
using MLDatasets
using Images
using ImageTransformations
using Interpolations
using LinearAlgebra

function resize_and_flatten_images(images)
    resized_flattened_images = []
    for i in 1:size(images, 3)
        img = images[:, :, i]
        img_resized = imresize(img, (7, 7); method=Lanczos4OpenCV())
        img_resized = Float32.(img_resized .> 0.0)
        img_flattened = vec(img_resized)  # Flatten the resized image
        push!(resized_flattened_images, img_flattened)
    end
    return resized_flattened_images
end

function load_and_resize_mnist(
    filter_digit::Union{Nothing,Int}=nothing,
    max_samples::Union{Nothing,Int}=nothing,
    normalize::Bool=false,
)
    # Load MNIST dataset
    train_x, train_y = MLDatasets.CIFAR100.traindata()
    test_x, test_y = MLDatasets.CIFAR100.testdata()

    # Filter by digit if specified
    if filter_digit !== nothing
        indices = findall(x -> x == filter_digit, train_y)
        train_x = train_x[:, :, indices]
        train_y = train_y[indices]
    end

    # Limit the number of samples if specified
    if max_samples !== nothing && max_samples <= size(train_x, 3)
        train_x = train_x[:, :, 1:max_samples]
        train_y = train_y[1:max_samples]
    end

    # Normalize data if specified
    if normalize
        train_x = @.(2.0f0 * Float32(train_x) - 1.0f0)
    else
        train_x = Float32.(train_x)
    end

    # Resize and flatten images
    resized_flattened_train_x = resize_and_flatten_images(train_x)

    # Convert the list of flattened images into a 2D array
    flattened_train_x_array = hcat(resized_flattened_train_x...)

    return flattened_train_x_array, train_y
end

function load_mnist_data(
    filter_digit::Union{Nothing,Int}=nothing,
    max_samples::Union{Nothing,Int}=nothing,
    normalize::Bool=false,
)
    # Load MNIST dataset
    train_x, train_y = MLDatasets.MNIST.traindata()
    test_x, test_y = MLDatasets.MNIST.testdata()

    # Filter by digit if specified
    if filter_digit !== nothing
        indices = findall(x -> x == filter_digit, train_y)
        train_x = train_x[:, :, indices]
        train_y = train_y[indices]
    end

    # Limit the number of samples if specified
    if max_samples !== nothing && max_samples <= size(train_x, 3)
        train_x = train_x[:, :, 1:max_samples]
        train_y = train_y[1:max_samples]
    end

    # Normalize data if specified
    if normalize
        train_x = @.(2.0f0 * Float32(train_x) - 1.0f0)
    else
        train_x = Float32.(train_x)
    end

    train_x = reshape(train_x, 28 * 28, :)
    return train_x, train_y
end

function load_mnist()
    # Load MNIST data
    train_x, train_y = MLDatasets.CIFAR100.traindata()
    test_x, test_y = MLDatasets.MNIST.testdata()

    return (reshape(Float32.(train_x), 32 * 32, :), train_y)#, (test_x, test_y)
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

(train_x, train_y) = load_and_resize_mnist(nothing, 60000, false)

(train_x, train_y) = load_mnist_data(nothing, 60000, false)

(train_x, train_y) = load_data(hparams)
(train_x, train_y) = (train_x[:, 1:5000], train_y[1:5000])
(train_x, train_y) = load_mnist(0)
(train_x, train_y) = load_mnist(9, 5000)
(train_x, train_y) = load_mnist_normalized(8, 100)

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

model = gpu(
    Chain(
        Dense(dims, 128, elu),
        Dense(128, 256, elu),
        Dense(256, 28 * 28, identity),
        Dropout(0.2),
    ),
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

function Generator_basic(latent_dim::Int)
    return Chain(
        Dense(latent_dim, 128, relu),
        Dense(128, 258, relu),
        Dense(258, 512, relu),
        Dense(512, 7 * 7, tanh),
        Dropout(0.05),
        Flux.flatten,
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

device = gpu

gen = device(Generator(latent_dim))
#model = Chain( ConvTranspose((7, 7), 100 => 256, stride=1, padding=0), BatchNorm(256, relu), ConvTranspose((4, 4), 256 => 128, stride=2, padding=1), BatchNorm(128, relu), ConvTranspose((4, 4), 128 => 1, stride=2, padding=1), tanh ))

# Mean vector (zero vector of length dim)
mean_vector = Float32.(zeros(latent_dim))

# Covariance matrix (identity matrix of size dim x dim)
cov_matrix = Float32.(Diagonal(ones(latent_dim)))

# Create the multivariate normal distribution
noise_model = device(MvNormal(mean_vector, cov_matrix))

n_samples = 1000

hparams = device(
    HyperParamsSlicedISL(;
        K=10, samples=1000, epochs=60, η=1e-3, noise_model=noise_model, m=10
    ),
)

# Create a data loader for training
batch_size = 1000
#train_loader = DataLoader(train_x; batchsize=batch_size, shuffle=false, partial=false)
train_loader = DataLoader(train_x; batchsize=batch_size, shuffle=true, partial=false)

total_loss = []
@showprogress for _ in 1:10
    append!(
        total_loss,
        sliced_invariant_statistical_loss_optimized_gpu_3(gen, train_loader, hparams),
        #sliced_invariant_statistical_loss_optimized_gpu_2(model, train_loader, hparams),
    )
end

img = cpu(gen(Float32.(rand(hparams.noise_model, 1))))
#img = @. (img + 1.0f0) / 2.0f0
img2 = reshape(img, 28, 28)
#display(Gray.(img2))
MLDatasets.MNIST.convert2image(img2)
transformed_matrix = Float32.(img2 .> 0.0)
display(Gray.(img2))

###DCGAN

using Base.Iterators: partition
using Flux
using Flux.Optimise: update!
using Flux: logitbinarycrossentropy, binarycrossentropy
using Images
using MLDatasets
using Statistics
using Parameters: @with_kw
using Random
using Printf
#using CUDAapi
using Zygote
if has_cuda()# Check if CUDA is available
    @info "CUDA is on"
    using CuArrays: CuArrays# If CUDA is available, import CuArrays
    CuArrays.allowscalar(false)
end

@with_kw struct HyperParams
    batch_size::Int = 128
    latent_dim::Int = 100
    epochs::Int = 10000
    verbose_freq::Int = 1000
    output_x::Int = 1       # No. of sample images to concatenate along x-axis
    output_y::Int = 1       # No. of sample images to concatenate along y-axis
    lr_dscr::Float32 = 0.0001
    lr_gen::Float32 = 0.0001
end

function generator(args)
    return gpu(
        Chain(
            Dense(args.latent_dim, 256),
            x -> leakyrelu.(x, 0.2f0),
            Dense(256, 512),
            x -> leakyrelu.(x, 0.2f0),
            Dense(512, 1024),
            x -> leakyrelu.(x, 0.2f0),
            Dense(1024, 32 * 32 * 3),  # Adjust the output size for 3 channels
            x -> reshape(x, 32, 32, 3, :),  # Reshape to 32x32 with 3 channels
            x -> tanh.(x),  # Apply the tanh activation function
        ),
    )
end

function discriminator(args)
    return gpu(
        Chain(
            Dense(32 * 32 * 3, 1024),  # Adjust the input size for 3 channels
            x -> leakyrelu.(x, 0.2f0),
            Dropout(0.3),
            Dense(1024, 512),
            x -> leakyrelu.(x, 0.2f0),
            Dropout(0.3),
            Dense(512, 256),
            x -> leakyrelu.(x),
            Dropout(0.3),
            Dense(256, 1, sigmoid),
        ),
    )
end

function load_data()
    # Load MNIST dataset
    train_x, train_y = MLDatasets.CIFAR100.traindata(Float32)
    # Normalize to [-1, 1] and convert it to WHCN
    #train_x = reshape(@.(2.0f0 * train_x - 1.0f0), 32, 32, :)
    #image_tensor = reshape(image_tensor, :, size(image_tensor, 4))
    # Partition into batches
    #data = [gpu(image_tensor[:, r]) for r in partition(1:60000, hparams.batch_size)]
    #return image_tensor
    return (reshape(Float32.(train_x), 32 * 32 * 3, :), train_y)#, (test_x, test_y)
end

# Loss functions
function discr_loss(real_output, fake_output)
    real_loss = mean(binarycrossentropy.(real_output, 1.0f0))
    fake_loss = mean(binarycrossentropy.(fake_output, 0.0f0))
    return (real_loss + fake_loss)
end

generator_loss(fake_output) = mean(binarycrossentropy.(fake_output, 1.0f0))

function train_discr(discr, original_data, fake_data, opt_discr)
    ps = Flux.params(discr)
    loss, back = Zygote.pullback(ps) do
        discr_loss(discr(original_data), discr(fake_data))
    end
    grads = back(1.0f0)
    update!(opt_discr, ps, grads)
    return loss
end

Zygote.@nograd train_discr

function train_gan(gen, discr, original_data, opt_gen, opt_discr, hparams)
    noise = gpu(randn!(similar(original_data, (hparams.latent_dim, hparams.batch_size))))
    loss = Dict()
    ps = Flux.params(gen)
    loss["gen"], back = Zygote.pullback(ps) do
        fake_ = gen(noise)
        loss["discr"] = train_discr(discr, original_data, fake_, opt_discr)
        generator_loss(discr(fake_))
    end
    grads = back(1.0f0)
    update!(opt_gen, ps, grads)
    return loss
end

function create_output_image(gen, fixed_noise, hparams)
    @eval Flux.istraining() = false
    fake_images = @. cpu(gen(fixed_noise))
    #fake_images = @. reshape(fake_images, 32, 32, 1, size(fake_images, 2))
    @eval Flux.istraining() = true

    # Scale the images to be within the range [0, 1]
    #
    #image_array = dropdims(
    #    reduce(vcat, reduce.(hcat, partition(fake_images, hparams.output_y))); dims=(3, 4)
    #)
    #image_array = reduce(vcat, reduce.(hcat, partition(fake_images, hparams.output_y)))
    #image_array = @. clamp(image_array, 0.0f0, 1.0f0)
    fake_images[1] = @. clamp(fake_images[1], 0.0f0, 1.0f0)
    #image_array = @. image_array + 1.0f0 / 2.0f0
    #p = imresize(image_array, (224, 224))
    #MLDatasets.CIFAR100.convert2image(image_array)
    return fake_images
end

function train()
    hparams = HyperParams()

    data = gpu(load_data()[1])
    data = DataLoader(data; batchsize=batch_size, shuffle=true, partial=false)

    fixed_noise = [
        gpu(randn(hparams.latent_dim, 1)) for _ in 1:(hparams.output_x * hparams.output_y)
    ]

    # Discriminator
    dscr = gpu(discriminator(hparams))

    # Generator
    gen = gpu(generator(hparams))
    #gen = model

    # Optimizers
    opt_dscr = ADAM(hparams.lr_dscr)
    opt_gen = ADAM(hparams.lr_gen)

    isdir("output_1") || mkdir("output_1")

    # Training
    train_steps = 0
    for ep in 1:(hparams.epochs)
        @info "Epoch $ep"
        for x in data
            # Update discriminator and generator
            loss = train_gan(gen, dscr, x, opt_gen, opt_dscr, hparams)

            if train_steps % hparams.verbose_freq == 0
                @info(
                    "Train step $(train_steps), Discriminator loss = $(loss["discr"]), Generator loss = $(loss["gen"])"
                )
                # Save generated fake image

                fake_images = @. cpu(gen(fixed_noise))
                fake_images = @. reshape(fake_images, 32, 32, 1, size(fake_images, 2))
                #println(countmap(onecold.(model.(reshape.(fake_images, 784)))))
                output_image = create_output_image(gen, fixed_noise, hparams)
                save(@sprintf("output_1/gan_steps_%06d.png", train_steps), output_image)
            end
            train_steps += 1
        end
    end

    output_image = create_output_image(gen, fixed_noise, hparams)
    return gen
end

gen = train()

p = 0
for i in 1:100
    hparams = HyperParams()

    fixed_noise = [
        gpu(randn(hparams.latent_dim, 1)) for _ in 1:(hparams.output_x * hparams.output_y)
    ]

    fake_images = @. cpu(gen(fixed_noise))
    fake_images = @. reshape(fake_images, 28, 28, 1, size(fake_images, 2))
    #values_array = collect(values(countmap(onecold.(model.(reshape.(fake_images, 784))))))
    #is_uniformly_distributed(values_array)

    #output_image = create_output_image(gen, fixed_noise, hparams)
    #=
    img2 = reshape.(fake_images, 28, 28)
    #display.(Gray.(img2))
    p = MLDatasets.MNIST.convert2image.(img2)
    # Assuming p is your Vector of ReinterpretArray{Gray{Float32}, 2, Float32, Matrix{Float32}, true}
    num_images = length(p)
    img_height, img_width = size(p[1])  # Assuming all images have the same size

    # Resize each image to 224x224
    resized_images = [imresize(Gray.(p[i]), (224, 224)) for i in 1:num_images]

    # Determine the grid size (number of rows and columns)
    grid_size = ceil(Int, sqrt(num_images))

    # Initialize an empty array for the mosaic image
    mosaic_height = grid_size * 224
    mosaic_width = grid_size * 224
    mosaic_image = fill(Gray{Float32}(0.0), mosaic_height, mosaic_width)

    # Place each resized image in the mosaic
    for i in 1:num_images
        row = div(i - 1, grid_size)
        col = mod(i - 1, grid_size)
        mosaic_image[(row * 224 + 1):((row + 1) * 224), (col * 224 + 1):((col + 1) * 224)] = resized_images[i]
    end
    plot(
        heatmap(
            mosaic_image;
            color=:grays,
            legend=false,
            xticks=false,
            yticks=false,
            frame=false,
        ),
    )
    =#

    countmap(onecold.(model.(reshape.(fake_images, 784))))
    values_array = collect(values(countmap(onecold.(model.(reshape.(fake_images, 784))))))

    p += is_uniformly_distributed(values_array)
end

using HypothesisTests

function is_uniformly_distributed(vector)
    # Transform the vector to the range [0, 1]
    min_val, max_val = minimum(vector), maximum(vector)
    normalized_vector = (vector .- min_val) ./ (max_val - min_val)

    # Perform the Kolmogorov-Smirnov test
    result = HypothesisTests.ApproximateOneSampleKSTest(normalized_vector, Uniform(0, 1))
    return pvalue(result)  # Typically, p-value > 0.05 means we do not reject the null hypothesis
end
