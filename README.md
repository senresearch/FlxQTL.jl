# flxQTL
## *flex*ible Multivariate Linear Mixed Model based *QTL* Analysis for Structured Multiple Traits 

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://hkim89.github.io/flxQTL.jl/stable) -->
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://hkim89.github.io/flxQTL.jl/dev)
[![Build Status](https://travis-ci.com/hkim89/flxQTL.jl.svg?branch=master)](https://travis-ci.org/github/hkim89/flxQTL.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/hkim89/flxQTL.jl?svg=true)](https://ci.appveyor.com/project/hkim89/flxQTL-jl)
[![Coverage](https://codecov.io/gh/hkim89/flxQTL.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/hkim89/flxQTL.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

*flxQTL.jl* is a a package for a multivariate linear mixed model based QTL analysis tool that supports 1D-, 2D-genome scans, 
[matplotlib]((http://matplotlib.org/)) based visualization using [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl) for 1D genome scan and effects plots, 
a heat map for 2D scan, computation of kinship matrices, and auxillary functions for standardization of trait data using the Huber loss, transformation of LOD scores to ``\\log_{10}P``, 
reording genotype data, etc.  

## Installation

The package can installed in following ways.
In a Julia REPL, press `]` to enter a package mode,

```julia
julia> ]
pkg> add flxQTL
```

Or, equivalently, 

```julia
julia> using Pkg; Pkg.add("flxQTL")
```
Currently Julia `1.5` supports for the package.


To remove the package from the Julia REPL,

```julia
julia> ] 
pkg> rm flxQTL
```
Equivalently,

```julia
julia> using Pkg; Pkg.rm("flxQTL")
```

## Choice of BLAS vendors

The package can be run in either `openblas` (built-in Julia dense linear algebra routines) or `MKL` (intel MKL linear algebra).  
Without the intel MKL hardware, the installation of *MKL.jl* in Julia can slightly improve the performance.  
For its installation, consult with [MKL.jl](https://github.com/JuliaComputing/MKL.jl).

## Documentation

The documentation is published in [flxQTL manual](https://hkim89.github.io/flxQTL.jl/stable).  To learn more about Julia functionality, the [Julia documentation](https://julialang.org) can be found.