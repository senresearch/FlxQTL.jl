# Package Guide


## Installation

The package can installed in following ways.
In a Julia REPL, press `]` to enter a package mode,

```julia
pkg> add FlxQTL
```

Or, equivalently, 

```julia
julia> using Pkg; Pkg.add("FlxQTL")
```
Currently Julia `1.5` supports for the package.


To remove the package from the Julia REPL,

```julia
pkg> rm FlxQTL
```
Equivalently,

```julia
julia> using Pkg; Pkg.rm("FlxQTL")
```


## Choice of BLAS vendors 

The package can be run in either `openblas` (built-in Julia dense linear algebra routines) or `MKL` (intel MKL linear algebra).  
Without the intel MKL hardware, the installation of *MKL.jl* in Julia can slightly improve the performance.  
For its installation, consult with [MKL.jl](https://github.com/JuliaComputing/MKL.jl).

