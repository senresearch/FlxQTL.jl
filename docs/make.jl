using FlxQTL
using Documenter

makedocs(;
    modules=[FlxQTL],
    authors="Hyeonju Kim <hyeonjukm01@gmail.com>",
#     repo="https://github.com/hkim89/flxQTL.jl/blob/{commit}{path}#L{line}",
    sitename="FlxQTL.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://hkim89.github.io/FlxQTL.jl/stable",
        assets=String[],
    ),
#     doctest=false,
    pages=[
        "Home" => "index.md",
        "Installation" => "guide/tutorial.md",
        "QTL Analysis" =>  "guide/analysis.md",
        "Types and Functions" => "functions.md"
    ],
)

deploydocs(;
    repo="github.com/hkim89/FlxQTL.jl.git",
    devurl="stable",
)
