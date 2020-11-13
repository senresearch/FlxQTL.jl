using flxQTL
using Documenter

makedocs(;
    modules=[flxQTL],
    authors="Hyeonju Kim <hyeonjukm01@gmail.com>",
    repo="https://github.com/hkim89/flxQTL.jl/blob/{commit}{path}#L{line}",
    sitename="flxQTL.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://hkim89.github.io/flxQTL.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/hkim89/flxQTL.jl",
)
