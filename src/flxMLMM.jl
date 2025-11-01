
"""

    flxMLMM


A module designed for fitting a Multivariate Linear Mixed Model
run by the Nesterov's Accelerated Gradient with restarting scheme embedding the Expectation Conditional Maximization to 
estimate MLEs.  REML is not supported.  The FlxQTL model is defined as 

```math
vec(Y)\\sim MVN((X' \\otimes Z)vec(B) (or ZBX), K \\otimes \\Omega +I \\otimes \\Sigma),
``` 

where `K` is a genetic kinship, and ``\\Omega \\approx \\tau^2V_C``, ``\\Sigma`` are covariance matrices for random and error terms, respectively.  
``V_C`` is pre-estimated under the null model (`H0`) of no QTL from the conventional MLMM, which is equivalent to the FlxQTL model for ``\\tau^2 =1``.  
``Z \\neq I_m`` estimates much smaller `B` than the former model with `Z = I`, where 
dim(Y) = (m traits, n individuals), and dim(X) = (p markers, n), dim(Z) = (m, q trait covariates).

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
