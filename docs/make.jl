using Documenter
using ISL

println("Generating Docs")

makedocs(;
    sitename = "ISL",
    format = Documenter.HTML(),
    modules = [ISL],
    pages=[
        "home" => "index.md",
        "GAN" => "gan.md",
        "Example" => "example.md",
        "Benchmark" => "benchmark.md",
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/josemanuel22/ISL.git",
    target="build",
    push_preview = true,
    devbranch="main",
)
