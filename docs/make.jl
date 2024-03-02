using Documenter
using ISL

println("Generating Docs")

makedocs(;
    sitename="ISL",
    format=Documenter.HTML(; assets=["./assets/isl.ico"]),
    modules=[ISL],
    authors="José Manuel de Frutos",
    pages=[
        "Home" => "index.md",
        "GANs" => "Gans.md",
        "Example" => "Examples.md",
        "DeepAR" => "DeepAR.md",
    ],
    checkdocs=:none,
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(;
    repo="github.com/josemanuel22/ISL.git",
    target="build",
    push_preview=true,
    devbranch="main",
)
