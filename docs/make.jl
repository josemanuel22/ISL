using Documenter
using AdaptativeBlockLearning

println("Generating Docs")

makedocs(
    sitename = "AdaptativeBlockLearning",
    format = Documenter.HTML(),
    modules = [AdaptativeBlockLearning]
    pages=[
        "Home" => "index.md",
        "GAN" => "GAN.md",
        "Example" => "example.md",
        "Benchmark Utils" => "benchmark_utils.md",
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/josemanuel22/AdaptativeBlockLearning.git",
    target="build",
    push_preview = true,
)
