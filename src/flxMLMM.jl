
"""

    flxMLMM


A module designed for fitting a Multivariate Linear Mixed Model
run by the Nesterov's Accelerated Gradient with restarting scheme embedding the Expectation Conditional Maximization to 
estimate MLEs.  REML is not supported.

The general form of Multivariate Linear Mixed model is 

```math
vec(Y) \sim MVN((X' \otimes Z)vec(B) (or ZBX),  K \otimes \V_C +I \otimes \Sigma),
```
where ``Z = I_m``, `K` is a genetic kinship, and ``V_C, \\Sigma`` are variance component and error matrices, respectively.  

The FlxQTL model (flxMLMM) estimates a scalar parameter ``\\tau^2`` under H1 to efficiently estimate the high dimensional variance 
component, i.e. ``\\Omega \\approx \\tau^2 V_C`` as well as ``Z \\neq I_m`` to estimate much smaller `B` than the former model with `Z = I`.
dim(Y) = (m traits, n individuals), dim(X) = (p markers, n), dim(Z) = (m, q trait covariates).

"""
module flxMLMM

#  __precompile__(true)
using Random
using LinearAlgebra, Distributed
import StatsBase: sample
import Statistics: mean, var, quantile,cov

using ..MLM
using ..GRM:kinshipLin, kinshipStd

using ..EcmNestrv:ecmLMM,ecmNestrvAG,NestrvAG,Approx,Result,updateΣ
# using ..EcmNestrv

using ..Util: mat2array,array2mat, Markers, newMarkers,lod2logP

include("QTLfunctions.jl")
# include("Miscellanea.jl")


end
