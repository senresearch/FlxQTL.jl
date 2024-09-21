
"""

    flxMLMM


A module designed for fitting a Multivariate Linear Mixed Model
run by Nesterov's Accelerated Gradient  with restarting scheme incorporated with Expectation Conditional Maximization.

The model:

``Y=XBZ'+R+E``, where ``E(vec(Y))= (Z \\otimes X)vec(B)``,  ``var(vec(Y))= \\tau^2 K_G \\otimes K_C + I_n \\otimes \\Sigma``

"""
module flxMLMM

#  __precompile__(true)
using Random
using LinearAlgebra, Distributed, DelimitedFiles
import StatsBase: sample, shuffle, cov
import Statistics: mean, var, quantile

using ..MLM

using ..EcmNestrv:ecmLMM,ecmNestrvAG,NestrvAG,Approx,Result,updateΣ

using ..Util: mat2array,array2mat, Markers, newMarkers,lod2logP

include("QTLfunctions.jl")
# include("Miscellanea.jl")


end
