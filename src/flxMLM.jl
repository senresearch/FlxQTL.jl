"""

    flxMLM


A module designed for fitting a Multivariate Linear Model by the (Residual) Maximum Likelihood (REML or MLE) method.  
The model:

``Y=XBZ'+E``, where ``E(vec(Y))= (Z \\otimes X)vec(B)``,  ``var(vec(Y))=  I_n \\otimes \\Sigma``

"""
module flxMLM

using LinearAlgebra, Distributed, Random
import Statistics: quantile

using ..MLM:mGLM, Estimat
using ..Util:lod2logP, Markers

include("geneScan.jl")
    


end