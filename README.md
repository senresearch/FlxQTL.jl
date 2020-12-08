# flxQTL

## *flex*ible Multivariate Linear Mixed Model based *QTL* Analysis for Structured Multiple Traits 

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://hkim89.github.io/flxQTL.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://hkim89.github.io/flxQTL.jl/dev)
[![Build Status](https://travis-ci.com/hkim89/flxQTL.jl.svg?branch=master)](https://travis-ci.org/github/hkim89/flxQTL.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/hkim89/flxQTL.jl?svg=true)](https://ci.appveyor.com/project/hkim89/flxQTL-jl)
[![Coverage](https://codecov.io/gh/hkim89/flxQTL.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/hkim89/flxQTL.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

*flxQTL.jl* is a a package for a multivariate linear mixed model based
QTL analysis tool that supports incorporating information from trait
covariates such as time or different environments.  The package
supports computation of one-dimensional and two-dimensional
multivariate genome scans, visualization of genome scans, support for
LOCO (leave-one-chromosome-out), computation of kinship matrices, and
support for distributed computing.

<!-- and auxillary functions for standardization of
trait data using the Huber loss, transformation of LOD scores to
``\\log_{10}P``,
reording genotype data, etc.  
-->

The package is written in [Julia](https://www.julialang.org) and
includes extensive
[documentation](https://hkim89.github.io/flxQTL.jl/stable).  If you
are new to Julia you may want to learn more by looking at [Julia
documentation](https://julialang.org).  Example data sets are located
in the [data](https://github.com/hkim89/flxQTL.jl/tree/master/data)
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

