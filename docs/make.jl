using Documenter
using AdaptativeBlockLearning

println("Generating Docs")

makedocs(
    sitename = "AdaptativeBlockLearning",
    format = Documenter.HTML(),
    modules = [AdaptativeBlockLearning]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/josemanuel22/AdaptativeBlockLearning.git",
)
