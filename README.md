# FlxQTL

## *Fl*e*x*ible Multivariate Linear Mixed Model based *QTL* Analysis for Structured Multiple Traits

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://senresearch.github.io/FlxQTL.jl/stable)
[![CI](https://github.com/senresearch/FlxQTL.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/senresearch/FlxQTL.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/senresearch/FlxQTL.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/senresearch/FlxQTL.jl)

*FlxQTL.jl* is a package for a multivariate linear mixed model based
QTL analysis tool that supports incorporating information from trait
covariates such as time or different environments.  The package
supports computation of one-dimensional and two-dimensional
multivariate genome scans, visualization of genome scans, support for
LOCO (leave-one-chromosome-out), computation of kinship matrices, and
support for distributed computing.

![1D Genome Scan](images/ex1.png)

![2D Genome Scan](images/ex2.jpg)

The package is written in [Julia](https://www.julialang.org) and
includes extensive
[documentation](https://senresearch.github.io/FlxQTL.jl/stable).  If you
are new to Julia you may want to learn more by looking at [Julia
documentation](https://julialang.org).  Example data sets are located
in the [data](https://github.com/senresearch/FlxQTL.jl/tree/master/data)
directory.  For details about the method, you may want to read our
paper available as a
[preprint](https://doi.org/10.1101/2020.03.27.012690).




## Paper

Flexible multivariate linear mixed models for structured multiple
traits
Hyeonju Kim, Gregory Farage, John T. Lovell, John K. Mckay, Thomas
E. Juenger, Śaunak Sen
doi: https://doi.org/10.1101/2020.03.27.012690

## Installation

The package can installed in following ways.
In a Julia REPL, press `]` to enter a package mode,

```julia
julia> ]
pkg> add FlxQTL
```

Or, equivalently,

```julia
julia> using Pkg; Pkg.add("FlxQTL")
```

For installing from the source,
```julia
pkg> add https://github.com/senresearch/FlxQTL.jl
```
or,

```julia
julia> Pkg.add(url="https://github.com/senresearch/FlxQTL.jl")
```

To remove the package from the Julia REPL,

```julia
julia> ]
pkg> rm FlxQTL
```
Equivalently,

```julia
julia> using Pkg; Pkg.rm("FlxQTL")
```

## Choice of BLAS vendors

The package can be run with OpenBLAS (built-in Julia dense linear
algebra routines) or MKL (Intel's Math Kernel Library).  `MKL.jl`
works best on Intel hardware, but it can slightly improve performance
without Intel hardware.  For installation and details,
see: [MKL.jl](https://github.com/JuliaComputing/MKL.jl).
